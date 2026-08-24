// Application shell: `winit` replaces SDL's window + event pump. This module owns the game
// state (equivalent of the SDL version's AppState, minus the canvas), maps input events onto
// players, and drives the per-frame simulate -> build scene -> record commands -> present path.

use std::{sync::Arc, time::Instant};

use vulkano::{
    Validated, VulkanError,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter},
    pipeline::graphics::viewport::Viewport,
    swapchain::{Surface, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image},
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{CursorGrabMode, Window, WindowId},
};

use crate::game::{
    LOOK_SENSITIVITY, MAX_PLAYER_COUNT, Player, init_players, shoot, update, whose_keyboard,
    whose_mouse,
};
use crate::map::{MAP_BOX_EDGES_LEN, MAP_BOX_SCALE, init_edges};
use crate::renderer::{
    Gpu, PushConstants, RenderContext, create_pipeline, create_render_pass, create_swapchain,
    window_size_dependent_setup,
};
use crate::scene::build_scene;

pub(crate) struct App {
    gpu: Gpu,

    // Game state (equivalent of the SDL version's AppState, minus the canvas)
    player_count: usize,
    players: [Player; MAX_PLAYER_COUNT],
    edges: [[f32; 6]; MAP_BOX_EDGES_LEN],

    last_frame: Option<Instant>,
    rcx: Option<RenderContext>,
}

impl App {
    pub(crate) fn new(event_loop: &EventLoop<()>) -> Self {
        let gpu = Gpu::new(event_loop);

        let mut players = [Player {
            mouse: None,
            keyboard: None,
            pos: [0.0; 3],
            vel: [0.0; 3],
            yaw: 0,
            pitch: 0,
            radius: 0.0,
            height: 0.0,
            color: [0; 3],
            wasd: 0,
        }; MAX_PLAYER_COUNT];

        let mut edges = [[0.0; 6]; MAP_BOX_EDGES_LEN];

        init_players(&mut players, MAX_PLAYER_COUNT);
        init_edges(MAP_BOX_SCALE, &mut edges, MAP_BOX_EDGES_LEN);

        App {
            gpu,
            player_count: 1,
            players,
            edges,
            last_frame: None,
            rcx: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Example splitscreen shooter game")
                        .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0)),
                )
                .unwrap(),
        );
        let surface = Surface::from_window(&self.gpu.instance, &window).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = create_swapchain(&self.gpu.device, &surface, window_size.into());

        let render_pass = create_render_pass(&self.gpu.device, swapchain.image_format());

        let framebuffers = window_size_dependent_setup(&images, &render_pass);

        let (pipeline, pipeline_layout) = create_pipeline(&self.gpu.device, &render_pass);

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            min_depth: 0.0,
            max_depth: 1.0,
        };

        // FPS-style controls: grab and hide the cursor so relative motion can be tracked
        let _ = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
        window.set_cursor_visible(false);

        let previous_frame_end = Some(sync::now(self.gpu.device.clone()).boxed());

        self.last_frame = Some(Instant::now());
        self.rcx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            pipeline_layout,
            viewport,
            recreate_swapchain: false,
            previous_frame_end,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                if let Some(rcx) = self.rcx.as_mut() {
                    rcx.recreate_swapchain = true;
                }
            }
            WindowEvent::MouseInput {
                device_id, state, ..
            } => {
                // Any button press shoots (SDL's MouseButtonDown)
                if state == ElementState::Pressed && self.whose_mouse(device_id).is_none() {
                    self.claim_mouse(device_id);
                }
                if state == ElementState::Pressed
                    && let Some(index) = self.whose_mouse(device_id)
                {
                    shoot(index, &mut self.players, self.player_count);
                }
            }
            WindowEvent::KeyboardInput {
                device_id,
                event: KeyEvent {
                    logical_key, state, ..
                },
                ..
            } => {
                // Escape releases the mouse (SDL exits on KeyUp Escape)
                if matches!(logical_key, Key::Named(NamedKey::Escape))
                    && state == ElementState::Released
                {
                    event_loop.exit();
                    return;
                }

                if self.whose_keyboard(device_id).is_none() {
                    self.claim_keyboard(device_id);
                }

                if let Some(index) = self.whose_keyboard(device_id) {
                    let bit = match &logical_key {
                        Key::Character(c) => match c.to_lowercase().as_str() {
                            "w" => Some(1),
                            "a" => Some(2),
                            "s" => Some(4),
                            "d" => Some(8),
                            _ => None,
                        },
                        Key::Named(NamedKey::Space) => Some(16),
                        _ => None,
                    };

                    if let Some(bit) = bit {
                        if state == ElementState::Pressed {
                            self.players[index].wasd |= bit;
                        } else {
                            self.players[index].wasd &= !bit;
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.redraw();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        // Equivalent of SDL's MouseMotion handling: relative motion rotates the player.
        // Raw DeviceEvent::MouseMotion deltas are used instead of absolute cursor positions
        // (diffed against the last position): with the cursor locked (Wayland) absolute
        // positions stop updating, and when merely confined (X11) they clamp at the window
        // edges, which stalls the view mid-turn. Raw deltas keep flowing in both cases.
        let DeviceEvent::MouseMotion { delta } = event else {
            return;
        };

        if self.whose_mouse(device_id).is_none() {
            self.claim_mouse(device_id);
        }

        if let Some(index) = self.whose_mouse(device_id) {
            let player = &mut self.players[index];
            // Mouse right turns right, mouse down looks down (same signs as the SDL version),
            // clamped to +/-90 degrees of pitch to prevent over-rotation.
            let yaw_delta = (-delta.0 * LOOK_SENSITIVITY) as i32;
            player.yaw = player.yaw.wrapping_add(yaw_delta as u32);

            let pitch_delta = (delta.1 * LOOK_SENSITIVITY) as i32;
            player.pitch = player
                .pitch
                .saturating_sub(pitch_delta)
                .clamp(-0x40000000, 0x40000000);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(rcx) = self.rcx.as_ref() {
            rcx.window.request_redraw();
        }
    }
}

impl App {
    fn whose_mouse(&self, mouse: DeviceId) -> Option<usize> {
        whose_mouse(mouse, &self.players, self.player_count)
    }

    fn whose_keyboard(&self, keyboard: DeviceId) -> Option<usize> {
        whose_keyboard(keyboard, &self.players, self.player_count)
    }

    // Assigns an unseen device to the first free player slot, growing the active player count
    fn claim_mouse(&mut self, mouse: DeviceId) {
        if let Some(i) = (0..MAX_PLAYER_COUNT).find(|&i| self.players[i].mouse.is_none()) {
            self.players[i].mouse = Some(mouse);
            self.player_count = self.player_count.max(i + 1);
        }
    }

    fn claim_keyboard(&mut self, keyboard: DeviceId) {
        if let Some(i) = (0..MAX_PLAYER_COUNT).find(|&i| self.players[i].keyboard.is_none()) {
            self.players[i].keyboard = Some(keyboard);
            self.player_count = self.player_count.max(i + 1);
        }
    }

    fn redraw(&mut self) {
        let Some(rcx) = self.rcx.as_mut() else {
            return;
        };

        let now = Instant::now();
        let dt_ns = now
            .duration_since(self.last_frame.unwrap_or(now))
            .as_nanos() as u64;
        self.last_frame = Some(now);

        // Physics update, identical to the SDL version
        update(&mut self.players, self.player_count, dt_ns);

        let window_size = rcx.window.inner_size();

        // Do not draw when the screen size is zero (e.g. minimized window)
        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        rcx.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if rcx.recreate_swapchain {
            let (new_swapchain, new_images) = rcx
                .swapchain
                .recreate(&SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..rcx.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            rcx.swapchain = new_swapchain;
            rcx.framebuffers = window_size_dependent_setup(&new_images, &rcx.render_pass);
            rcx.viewport.extent = window_size.into();
            rcx.recreate_swapchain = false;
        }

        // Build the frame's line geometry on the CPU (clipping/projection like the SDL version)
        let mut vertices = Vec::new();
        let mut regions = Vec::new();
        build_scene(
            &self.edges,
            &self.players,
            self.player_count,
            window_size.width,
            window_size.height,
            &mut vertices,
            &mut regions,
        );

        // Upload the vertices; a fresh buffer per frame avoids any data races between frames
        // in flight.
        let vertex_buffer = Buffer::from_iter(
            &self.gpu.memory_allocator,
            &BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            &AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            self.gpu.command_buffer_allocator.clone(),
            self.gpu.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        rcx.framebuffers[image_index as usize].clone(),
                    )
                },
                Default::default(),
            )
            .unwrap()
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())
            .unwrap();

        builder
            .push_constants(
                rcx.pipeline_layout.clone(),
                0,
                PushConstants {
                    resolution: rcx.viewport.extent,
                },
            )
            .unwrap()
            .bind_pipeline_graphics(rcx.pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .unwrap();

        // One draw call per split-screen region; the scissor replaces SDL's clip rect
        for region in &regions {
            builder
                .set_scissor(0, [region.scissor].into_iter().collect())
                .unwrap();
            unsafe { builder.draw(region.vertex_count, 1, region.first_vertex, 0) }.unwrap();
        }

        builder.end_render_pass(Default::default()).unwrap();

        let command_buffer = builder.build().unwrap();
        let future = rcx
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.gpu.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.gpu.queue.clone(),
                SwapchainPresentInfo::new(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.gpu.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                rcx.previous_frame_end = Some(sync::now(self.gpu.device.clone()).boxed());
            }
        }
    }
}
