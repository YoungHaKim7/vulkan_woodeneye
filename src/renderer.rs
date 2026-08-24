// Vulkan rendering layer, the part that replaces the SDL canvas:
// - instance/physical-device/device/queue selection and the allocators (`Gpu`),
// - swapchain, render pass, and the line-list pipeline (`RenderContext` + `create_*`),
// - the GLSL shaders that map window-pixel coordinates to Vulkan NDC.

use std::sync::Arc;

use vulkano::{
    VulkanLibrary,
    buffer::BufferContents,
    command_buffer::allocator::StandardCommandBufferAllocator,
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    format::Format,
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    single_pass_renderpass,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::GpuFuture,
};
use winit::{event_loop::EventLoop, window::Window};

use crate::scene::LineVertex;

#[derive(BufferContents)]
#[repr(C)]
pub(crate) struct PushConstants {
    pub(crate) resolution: [f32; 2],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 450

            layout(push_constant) uniform Push {
                vec2 resolution;
            } pc;

            layout(location = 0) in vec2 position;
            layout(location = 1) in vec4 color;

            layout(location = 0) out vec4 v_color;

            void main() {
                // `position` is in window pixels with y pointing down, which is the same
                // direction as Vulkan's NDC y axis, so it maps across directly. Do NOT apply
                // the OpenGL-style y negation here: in Vulkan that flips the whole image
                // upside down (floor at the top, ceiling at the bottom).
                vec2 ndc = position / pc.resolution * 2.0 - 1.0;
                gl_Position = vec4(ndc.x, ndc.y, 0.0, 1.0);
                v_color = color;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) in vec4 v_color;
            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = v_color;
            }
        ",
    }
}

// Vulkan objects that live for the whole run, independent of any window.
pub(crate) struct Gpu {
    pub(crate) instance: Arc<Instance>,
    pub(crate) device: Arc<Device>,
    pub(crate) queue: Arc<Queue>,
    pub(crate) memory_allocator: Arc<StandardMemoryAllocator>,
    pub(crate) command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl Gpu {
    pub(crate) fn new(event_loop: &EventLoop<()>) -> Self {
        let library = unsafe { VulkanLibrary::new() }.unwrap();

        // All the window-drawing functionalities are part of non-core extensions that we need to
        // enable manually, so we ask `Surface` for the list of extensions required.
        let required_extensions = Surface::required_extensions(event_loop);

        let instance = Instance::new(
            &library,
            &InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: &required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("no suitable physical device found");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            &physical_device,
            &DeviceCreateInfo {
                enabled_extensions: &device_extensions,
                queue_create_infos: &[QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .unwrap();
        let queue = queues.next().unwrap();

        let memory_allocator = Arc::new(StandardMemoryAllocator::new(&device, &Default::default()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            &device,
            &Default::default(),
        ));

        Gpu {
            instance,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
        }
    }
}

// Per-window Vulkan objects, (re)created in `resumed` and on resize.
pub(crate) struct RenderContext {
    pub(crate) window: Arc<Window>,
    pub(crate) swapchain: Arc<Swapchain>,
    pub(crate) render_pass: Arc<RenderPass>,
    pub(crate) framebuffers: Vec<Arc<Framebuffer>>,
    pub(crate) pipeline: Arc<GraphicsPipeline>,
    pub(crate) pipeline_layout: Arc<PipelineLayout>,
    pub(crate) viewport: Viewport,
    pub(crate) recreate_swapchain: bool,
    pub(crate) previous_frame_end: Option<Box<dyn GpuFuture>>,
}

pub(crate) fn create_swapchain(
    device: &Arc<Device>,
    surface: &Arc<Surface>,
    window_size: [u32; 2],
) -> (Arc<Swapchain>, Vec<Arc<vulkano::image::Image>>) {
    let surface_capabilities = device
        .physical_device()
        .surface_capabilities(surface, &Default::default())
        .unwrap();
    let (image_format, _) = device
        .physical_device()
        .surface_formats(surface, &Default::default())
        .unwrap()[0];

    Swapchain::new(
        device,
        surface,
        &SwapchainCreateInfo {
            min_image_count: surface_capabilities.min_image_count.max(3),
            image_format,
            image_extent: window_size,
            image_usage: vulkano::image::ImageUsage::COLOR_ATTACHMENT,
            composite_alpha: surface_capabilities
                .supported_composite_alpha
                .into_iter()
                .next()
                .unwrap(),
            ..Default::default()
        },
    )
    .unwrap()
}

pub(crate) fn create_render_pass(device: &Arc<Device>, image_format: Format) -> Arc<RenderPass> {
    single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: image_format,
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .unwrap()
}

pub(crate) fn create_pipeline(
    device: &Arc<Device>,
    render_pass: &Arc<RenderPass>,
) -> (Arc<GraphicsPipeline>, Arc<PipelineLayout>) {
    let vs = unsafe { vs::load(device) }
        .unwrap()
        .entry_point("main")
        .unwrap();
    let fs = unsafe { fs::load(device) }
        .unwrap()
        .entry_point("main")
        .unwrap();
    let vertex_input_state = LineVertex::per_vertex().definition(&vs).unwrap();
    let stages = [
        PipelineShaderStageCreateInfo::new(&vs),
        PipelineShaderStageCreateInfo::new(&fs),
    ];
    let layout = PipelineLayout::from_stages(device, &stages).unwrap();
    let subpass = Subpass::new(render_pass, 0).unwrap();

    let pipeline = GraphicsPipeline::new(
        device,
        None,
        &GraphicsPipelineCreateInfo {
            stages: &stages,
            vertex_input_state: Some(&vertex_input_state),
            // Draw line segments instead of triangles
            input_assembly_state: Some(&InputAssemblyState {
                topology: PrimitiveTopology::LineList,
                ..Default::default()
            }),
            viewport_state: Some(&ViewportState::default()),
            rasterization_state: Some(&RasterizationState::default()),
            multisample_state: Some(&MultisampleState::default()),
            color_blend_state: Some(&ColorBlendState {
                attachments: &[ColorBlendAttachmentState::default()],
                ..Default::default()
            }),
            // Dynamic viewport + scissor: one draw call per split-screen region
            dynamic_state: &[DynamicState::Viewport, DynamicState::Scissor],
            subpass: Some((&subpass).into()),
            ..GraphicsPipelineCreateInfo::new(&layout)
        },
    )
    .unwrap();
    (pipeline, layout)
}

/// Called once during initialization, then again whenever the window is resized.
pub(crate) fn window_size_dependent_setup(
    images: &[Arc<vulkano::image::Image>],
    render_pass: &Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = vulkano::image::view::ImageView::new_default(image).unwrap();

            Framebuffer::new(
                render_pass,
                &FramebufferCreateInfo {
                    attachments: &[&view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect()
}
