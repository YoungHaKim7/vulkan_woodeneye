// Vulkan recreation of /a02_woodeneye_008.rs
// original code : https://github.com/libsdl-org/SDL/tree/main/examples/demo/02-woodeneye-008
//
// All of the game simulation code (players, physics, shooting, view math) is kept identical to
// the SDL version. Only the windowing/rendering layer differs:
// - `winit` replaces SDL's window + event pump,
// - Vulkan (via vulkano) replaces the SDL canvas.
//
// The original renders CPU-side clipped/projected 2D lines onto a canvas. Here the exact same
// clipping/projection math produces window pixel-space line segments that are uploaded as vertex
// data and rasterized with a line-list pipeline. Per-player split-screen viewports are done with
// dynamic scissors instead of SDL clip rectangles.
//
// Deliberate deviations from the SDL version's gameplay:
// - the map is a small cube (MAP_BOX_SCALE 4 instead of 16) and players spawn standing on the
//   floor instead of falling into a huge box from mid-height,
// - movement acceleration scales with the box size (see `update`),
// - mouse look consumes winit's relative DeviceEvent::MouseMotion deltas with the original
//   demo's sensitivity, instead of diffing absolute cursor positions.

use std::{sync::Arc, time::Instant};

use vulkano::{
    Validated, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo,
        allocator::StandardCommandBufferAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::PhysicalDeviceType,
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Scissor, Viewport, ViewportState},
        },
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    single_pass_renderpass,
    swapchain::{
        Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo, acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{Key, NamedKey},
    window::{CursorGrabMode, Window, WindowId},
};

// Constants defining map size, player count, and drawing precision.
// The original demo uses 16; 4 makes a small cube you can walk around in.
const MAP_BOX_SCALE: i32 = 4;
const MAP_BOX_EDGES_LEN: usize = 12 + (MAP_BOX_SCALE * 2) as usize; // Number of map edges
const MAX_PLAYER_COUNT: usize = 4; // Maximum number of players
const CIRCLE_DRAW_SIDES: usize = 32; // Number of sides for drawing circles

// Mouse/keyboard rotation sensitivity, identical to the SDL version (per pixel of motion).
// This is 0x00080000 in the original C demo; the larger 0x00400000 made look 8x too fast.
const LOOK_SENSITIVITY: f64 = 524_288.0; // 0x00080000

// Structure representing a player.
// The SDL version stores raw u32 device IDs; winit uses its own `DeviceId` type instead.
#[derive(Clone, Copy)]
struct Player {
    mouse: Option<DeviceId>,    // ID of the mouse associated with the player
    keyboard: Option<DeviceId>, // ID of the keyboard associated with the player
    pos: [f64; 3],              // 3D position of the player (x, y, z)
    vel: [f64; 3],              // 3D velocity of the player (x, y, z)
    yaw: u32,                   // Horizontal rotation of the player (angle)
    pitch: i32,                 // Vertical rotation of the player (angle)
    radius: f32,                // Radius of the player's collision circle
    height: f32,                // Height of the player
    color: [u8; 3],             // RGB color of the player
    wasd: u8,                   // Bitmask representing WASD key presses (Up, Left, Down, Right)
}

// Function to find a player by their mouse ID
fn whose_mouse(mouse: DeviceId, players: &[Player], _players_len: usize) -> Option<usize> {
    players.iter().position(|p| p.mouse == Some(mouse))
}

// Function to find a player by their keyboard ID
fn whose_keyboard(keyboard: DeviceId, players: &[Player], _players_len: usize) -> Option<usize> {
    players.iter().position(|p| p.keyboard == Some(keyboard))
}

// Tiny xorshift64* PRNG; stands in for the `rand` crate used by the SDL version.
fn next_random_byte() -> u8 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0x9e3779b97f4a7c15);
    let mut x = STATE.load(Ordering::Relaxed);
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    STATE.store(x, Ordering::Relaxed);
    (x.wrapping_mul(0x2545f4914f6cdd1d) >> 56) as u8
}

// Function to handle shooting (simplified hit detection), unchanged from the SDL version
fn shoot(shooter: usize, players: &mut [Player], players_len: usize) {
    let x0 = players[shooter].pos[0]; // Shooter's x position
    let y0 = players[shooter].pos[1]; // Shooter's y position
    let z0 = players[shooter].pos[2]; // Shooter's z position

    // Convert yaw and pitch to radians
    let bin_rad = std::f64::consts::PI / 2147483648.0;
    let yaw_rad = bin_rad * (players[shooter].yaw) as f64;
    let pitch_rad = bin_rad * players[shooter].pitch as f64;

    // Calculate shooting direction vector
    let cos_yaw = yaw_rad.cos();
    let sin_yaw = yaw_rad.sin();
    let cos_pitch = pitch_rad.cos();
    let sin_pitch = pitch_rad.sin();
    let vx = -sin_yaw * cos_pitch;
    let vy = sin_pitch;
    let vz = -cos_yaw * cos_pitch;

    // Iterate through other players to check for hits
    for i in 0..players_len {
        if i == shooter {
            continue; // Skip the shooter themselves
        }
        let target = &mut players[i];
        let mut hit = 0; // Initialize hit counter for head and feet check
        for j in 0..2 {
            // Check head and feet
            let r = target.radius as f64; // Target's radius
            let h = target.height as f64; // Target's height
            let dx = target.pos[0] - x0; // Difference in x position
            let dy = target.pos[1] - y0 + if j == 0 { 0.0 } else { r - h }; // Head/feet offset
            let dz = target.pos[2] - z0; // Difference in z position
            let vd = vx * dx + vy * dy + vz * dz;
            let dd = dx * dx + dy * dy + dz * dz;
            let vv = vx * vx + vy * vy + vz * vz;
            let rr = r * r;

            // Simplified hit detection (cone intersection with player's bounding sphere)
            if vd < 0.0 {
                continue;
            }
            if vd * vd >= vv * (dd - rr) {
                hit += 1;
            }
        }
        if hit > 0 {
            // If hit, reset the target's position to a random location
            target.pos[0] = (MAP_BOX_SCALE as f64 * (next_random_byte() as f64 - 128.0)) / 256.0;
            target.pos[1] = (MAP_BOX_SCALE as f64 * (next_random_byte() as f64 - 128.0)) / 256.0;
            target.pos[2] = (MAP_BOX_SCALE as f64 * (next_random_byte() as f64 - 128.0)) / 256.0;
        }
    }
}

// Function to update player positions and velocities based on input and physics,
// unchanged from the SDL version (see the comments there for a full explanation).
fn update(players: &mut [Player], players_len: usize, dt_ns: u64) {
    let time = dt_ns as f64 * 1e-9; // Convert time difference to seconds
    for player in players.iter_mut().take(players_len) {
        let rate = 6.0; // Rate of drag
        let drag = (-time * rate).exp(); // Calculate drag factor
        let diff = 1.0 - drag; // Calculate difference factor
        // Movement multiplier. The SDL version uses 60.0 for its 16-unit box; scaling it with
        // the box size keeps the same feel inside the smaller cube (top speed = mult / rate).
        let mult = 60.0 * MAP_BOX_SCALE as f64 / 16.0;
        let grav = 25.0; // Gravity acceleration

        // Calculate player's direction based on yaw and WASD input
        let yaw = player.yaw as f64;
        let rad = yaw * std::f64::consts::PI / 2147483648.0;
        let cos = rad.cos();
        let sin = rad.sin();
        let wasd = player.wasd;

        // Determine direction of movement based on WASD keys
        let dir_x = if wasd & 8 != 0 { 1.0 } else { 0.0 } - if wasd & 2 != 0 { 1.0 } else { 0.0 };
        let dir_z = if wasd & 4 != 0 { 1.0 } else { 0.0 } - if wasd & 1 != 0 { 1.0 } else { 0.0 };
        let norm = dir_x * dir_x + dir_z * dir_z;

        // Calculate acceleration based on direction and multiplier
        let acc_x = mult
            * if norm == 0.0 {
                0.0
            } else {
                (cos * dir_x + sin * dir_z) / norm.sqrt()
            };
        let acc_z = mult
            * if norm == 0.0 {
                0.0
            } else {
                (-sin * dir_x + cos * dir_z) / norm.sqrt()
            };

        // Update player's velocity with drag and acceleration
        let vel_x = player.vel[0];
        let vel_y = player.vel[1];
        let vel_z = player.vel[2];

        player.vel[0] -= vel_x * diff; // Apply drag to x velocity
        player.vel[1] -= grav * time; // Apply gravity to y velocity
        player.vel[2] -= vel_z * diff; // Apply drag to z velocity

        player.vel[0] += diff * acc_x / rate; // Apply acceleration to x velocity
        player.vel[2] += diff * acc_z / rate; // Apply acceleration to z velocity

        // Update player's position based on velocity and acceleration
        player.pos[0] += (time - diff / rate) * acc_x / rate + diff * vel_x / rate;
        player.pos[1] += -0.5 * grav * time * time + vel_y * time;
        player.pos[2] += (time - diff / rate) * acc_z / rate + diff * vel_z / rate;

        // Keep player within map bounds
        let scale = MAP_BOX_SCALE as f64;
        let bound = scale - player.radius as f64;
        let pos_x = player.pos[0].max(-bound).min(bound);
        let pos_y = player.pos[1].max(player.height as f64 - scale).min(bound);
        let pos_z = player.pos[2].max(-bound).min(bound);

        // Handle collisions with map boundaries
        if player.pos[0] != pos_x {
            player.vel[0] = 0.0;
        }
        if player.pos[1] != pos_y {
            // Set y velocity if spacebar is pressed (jumping)
            player.vel[1] = if wasd & 16 != 0 { 8.4375 } else { 0.0 };
        }
        if player.pos[2] != pos_z {
            player.vel[2] = 0.0;
        }
        player.pos[0] = pos_x;
        player.pos[1] = pos_y;
        player.pos[2] = pos_z;
    }
}

fn init_players(players: &mut [Player], len: usize) {
    // Initialize player positions. Players are placed in a grid-like pattern.
    for i in 0..len {
        players[i].radius = 0.5;
        players[i].height = 1.5;

        // Spawn halfway between the center and each wall, standing on the floor: `update`
        // clamps y to height - scale, which is exactly the standing eye height, so nobody
        // starts floating in mid-air.
        let half = MAP_BOX_SCALE as f64 * 0.5;
        players[i].pos[0] = half * if i & 1 != 0 { -1.0 } else { 1.0 };
        players[i].pos[1] = players[i].height as f64 - MAP_BOX_SCALE as f64;
        players[i].pos[2] =
            half * if i & 1 != 0 { -1.0 } else { 1.0 } * if i & 2 != 0 { -1.0 } else { 1.0 };

        players[i].vel[0] = 0.0;
        players[i].vel[1] = 0.0;
        players[i].vel[2] = 0.0;

        // The bitwise operations distribute the players around the origin.
        players[i].yaw = 0x20000000
            + if i & 1 != 0 { 0x80000000 } else { 0 }
            + if i & 2 != 0 { 0x40000000 } else { 0 };

        players[i].pitch = -0x08000000;

        players[i].wasd = 0;

        players[i].mouse = None;
        players[i].keyboard = None;

        // Generate a variety of colors per player index (unchanged from the SDL version).
        players[i].color[0] = if (1 << (i / 2)) & 2 != 0 { 0 } else { 0xff };
        players[i].color[1] = if (1 << (i / 2)) & 1 != 0 { 0 } else { 0xff };
        players[i].color[2] = if (1 << (i / 2)) & 4 != 0 { 0 } else { 0xff };

        players[i].color[0] = if i & 1 != 0 {
            players[i].color[0]
        } else {
            !players[i].color[0]
        };
        players[i].color[1] = if i & 1 != 0 {
            players[i].color[1]
        } else {
            !players[i].color[1]
        };
        players[i].color[2] = if i & 1 != 0 {
            players[i].color[2]
        } else {
            !players[i].color[2]
        };
    }
}

fn init_edges(scale: i32, edges: &mut [[f32; 6]], _edges_len: usize) {
    let r = scale as f32;

    #[rustfmt::skip]
    let map = [
        0, 1, 1, 3, 3, 2, 2, 0, // First 4 edges (bottom face)
        7, 6, 6, 4, 4, 5, 5, 7, // Next 4 edges (top face)
        6, 2, 3, 7, 0, 4, 5, 1, // Last 4 edges (connecting top and bottom)
    ];

    // Initialize the first 12 edges (the cube's edges).
    for i in 0..12 {
        for j in 0..3 {
            edges[i][j] = if map[i * 2] & (1 << j) != 0 { r } else { -r };
            edges[i][j + 3] = if map[i * 2 + 1] & (1 << j) != 0 {
                r
            } else {
                -r
            };
        }
    }

    // Initialize the remaining edges (the "walls" extending outwards).
    for i in 0..scale as usize {
        let d = (i * 2) as f32;

        for j in 0..2 {
            edges[i + 12][3 * j] = if j != 0 { r } else { -r };
            edges[i + 12][3 * j + 1] = -r;
            edges[i + 12][3 * j + 2] = d - r;

            edges[i + 12 + scale as usize][3 * j] = d - r;
            edges[i + 12 + scale as usize][3 * j + 1] = -r;
            edges[i + 12 + scale as usize][3 * j + 2] = if j != 0 { r } else { -r };
        }
    }
}

/// One vertex of a line segment, in window pixel coordinates (y down), with a color.
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct LineVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    color: [u8; 4],
}

#[derive(BufferContents)]
#[repr(C)]
struct PushConstants {
    resolution: [f32; 2],
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

// A split-screen region: which scissors rectangle it occupies and where its vertices live.
struct RegionGeometry {
    scissor: Scissor,
    first_vertex: u32,
    vertex_count: u32,
}

// Port of the original `draw_clipped_segment`, minus the actual drawing: returns the projected
// 2D offsets from the viewport origin after clipping against the near plane z = -w.
fn project_clipped_segment(
    mut ax: f32,
    mut ay: f32,
    mut az: f32,
    mut bx: f32,
    mut by: f32,
    mut bz: f32,
    z: f32,
    w: f32,
) -> Option<([f32; 2], [f32; 2])> {
    // Both points behind the clipping plane: nothing to draw
    if az >= -w && bz >= -w {
        return None;
    }

    let dx = ax - bx;
    let dy = ay - by;

    // Clip the first point (A) if it's behind the clipping plane
    if az > -w {
        let t = (-w - bz) / (az - bz);
        ax = bx + dx * t;
        ay = by + dy * t;
        az = -w;
    }

    // Clip the second point (B) if it's behind the clipping plane
    if bz > -w {
        let t = (-w - az) / (bz - az);
        bx = ax - dx * t;
        by = ay - dy * t;
        bz = -w;
    }

    // Perspective projection: project the 3D points to 2D offsets
    Some(([-z * ax / az, -z * ay / az], [-z * bx / bz, -z * by / bz]))
}

// Builds all line-segment vertices for the current frame. This mirrors the original `draw`
// function: same viewport splitting, same view matrix, same clipping/projection, same colors.
fn build_scene(
    edges: &[[f32; 6]],
    players: &[Player],
    players_len: usize,
    win_w: u32,
    win_h: u32,
    vertices: &mut Vec<LineVertex>,
    regions: &mut Vec<RegionGeometry>,
) {
    vertices.clear();
    regions.clear();

    if players_len == 0 {
        return;
    }

    const GRAY: [u8; 4] = [64, 64, 64, 255];
    const WHITE: [u8; 4] = [255, 255, 255, 255];

    let wf = win_w as f32;
    let hf = win_h as f32;

    // Calculate how to split the screen based on the number of players
    let part_hor = if players_len > 2 { 2 } else { 1 };
    let part_ver = if players_len > 1 { 2 } else { 1 };
    let size_hor = wf / part_hor as f32;
    let size_ver = hf / part_ver as f32;

    for i in 0..players_len {
        let player = &players[i];

        let mod_x = (i % part_hor) as f32;
        let mod_y = (i / part_hor) as f32;
        let hor_origin = (mod_x + 0.5) * size_hor;
        let ver_origin = (mod_y + 0.5) * size_ver;
        let cam_origin = 0.5 * (size_hor * size_hor + size_ver * size_ver).sqrt();
        let hor_offset = mod_x * size_hor;
        let ver_offset = mod_y * size_ver;

        // SDL clip rect -> dynamic scissor rectangle
        let off_x = (hor_offset.round() as u32).min(win_w.saturating_sub(1));
        let off_y = (ver_offset.round() as u32).min(win_h.saturating_sub(1));
        let ext_x = (size_hor.round() as u32).clamp(1, win_w - off_x);
        let ext_y = (size_ver.round() as u32).clamp(1, win_h - off_y);

        let first_vertex = vertices.len() as u32;

        let x0 = player.pos[0];
        let y0 = player.pos[1];
        let z0 = player.pos[2];

        // Pre-calculate trigonometric values for player's view direction
        let bin_rad = std::f64::consts::PI / 2147483648.0;
        let yaw_rad = bin_rad * player.yaw as f64;
        let pitch_rad = bin_rad * player.pitch as f64;
        let cos_yaw = yaw_rad.cos();
        let sin_yaw = yaw_rad.sin();
        let cos_pitch = pitch_rad.cos();
        let sin_pitch = pitch_rad.sin();

        // Create the view matrix (combining rotation)
        let mat = [
            cos_yaw as f32,
            0.0,
            -sin_yaw as f32,
            sin_yaw as f32 * sin_pitch as f32,
            cos_pitch as f32,
            cos_yaw as f32 * sin_pitch as f32,
            sin_yaw as f32 * cos_pitch as f32,
            -sin_pitch as f32,
            cos_yaw as f32 * cos_pitch as f32,
        ];

        // Draw each edge of the map (transformed exactly like the SDL version)
        for line in edges.iter() {
            let ax = mat[0] * (line[0] as f64 - x0) as f32
                + mat[1] * (line[1] as f64 - y0) as f32
                + mat[2] * (line[2] as f64 - z0) as f32;
            let ay = mat[3] * (line[0] as f64 - x0) as f32
                + mat[4] * (line[1] as f64 - y0) as f32
                + mat[5] * (line[2] as f64 - z0) as f32;
            let az = mat[6] * (line[0] as f64 - x0) as f32
                + mat[7] * (line[1] as f64 - y0) as f32
                + mat[8] * (line[2] as f64 - z0) as f32;
            let bx = mat[0] * (line[3] as f64 - x0) as f32
                + mat[1] * (line[4] as f64 - y0) as f32
                + mat[2] * (line[5] as f64 - z0) as f32;
            let by = mat[3] * (line[3] as f64 - x0) as f32
                + mat[4] * (line[4] as f64 - y0) as f32
                + mat[5] * (line[5] as f64 - z0) as f32;
            let bz = mat[6] * (line[3] as f64 - x0) as f32
                + mat[7] * (line[4] as f64 - y0) as f32
                + mat[8] * (line[5] as f64 - z0) as f32;

            if let Some((pa, pb)) = project_clipped_segment(ax, ay, az, bx, by, bz, cam_origin, 1.0)
            {
                // Convert to screen coordinates (same truncation as SDL Point::new)
                vertices.push(LineVertex {
                    position: [
                        (hor_origin + pa[0]) as i32 as f32,
                        (ver_origin - pa[1]) as i32 as f32,
                    ],
                    color: GRAY,
                });
                vertices.push(LineVertex {
                    position: [
                        (hor_origin + pb[0]) as i32 as f32,
                        (ver_origin - pb[1]) as i32 as f32,
                    ],
                    color: GRAY,
                });
            }
        }

        // Draw other players
        for j in 0..players_len {
            if i == j {
                continue; // Don't draw the current player
            }
            let target = &players[j];
            let color = [target.color[0], target.color[1], target.color[2], 255];

            // Draw the target player's top and bottom circles
            for k in 0..2u8 {
                let rx = target.pos[0] - player.pos[0];
                let ry = target.pos[1] - player.pos[1]
                    + (target.radius as f64 - target.height as f64) * k as f64;
                let rz = target.pos[2] - player.pos[2];

                let dx = mat[0] as f64 * rx + mat[1] as f64 * ry + mat[2] as f64 * rz;
                let dy = mat[3] as f64 * rx + mat[4] as f64 * ry + mat[5] as f64 * rz;
                let dz = mat[6] as f64 * rx + mat[7] as f64 * ry + mat[8] as f64 * rz;

                // If the target is behind the player, don't draw it
                if dz >= 0.0 {
                    continue;
                }

                let r_eff = target.radius as f64 * cam_origin as f64 / dz;
                let cx = hor_origin - cam_origin * dx as f32 / dz as f32;
                let cy = ver_origin + cam_origin * dy as f32 / dz as f32;

                // Circle drawn as a line loop of CIRCLE_DRAW_SIDES segments (SDL draw_lines)
                for s in 0..CIRCLE_DRAW_SIDES {
                    let a0 = 2.0 * std::f64::consts::PI * s as f64 / CIRCLE_DRAW_SIDES as f64;
                    let a1 = 2.0 * std::f64::consts::PI * (s + 1) as f64 / CIRCLE_DRAW_SIDES as f64;
                    vertices.push(LineVertex {
                        position: [
                            cx + r_eff as f32 * a0.cos() as f32,
                            cy + r_eff as f32 * a0.sin() as f32,
                        ],
                        color,
                    });
                    vertices.push(LineVertex {
                        position: [
                            cx + r_eff as f32 * a1.cos() as f32,
                            cy + r_eff as f32 * a1.sin() as f32,
                        ],
                        color,
                    });
                }
            }
        }

        // White crosshair at the center of this viewport
        vertices.push(LineVertex {
            position: [hor_origin as i32 as f32, (ver_origin - 10.0) as i32 as f32],
            color: WHITE,
        });
        vertices.push(LineVertex {
            position: [hor_origin as i32 as f32, (ver_origin + 10.0) as i32 as f32],
            color: WHITE,
        });
        vertices.push(LineVertex {
            position: [(hor_origin - 10.0) as i32 as f32, ver_origin as i32 as f32],
            color: WHITE,
        });
        vertices.push(LineVertex {
            position: [(hor_origin + 10.0) as i32 as f32, ver_origin as i32 as f32],
            color: WHITE,
        });

        regions.push(RegionGeometry {
            scissor: Scissor {
                offset: [off_x, off_y],
                extent: [ext_x, ext_y],
            },
            first_vertex,
            vertex_count: vertices.len() as u32 - first_vertex,
        });
    }
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    // Game state (equivalent of the SDL version's AppState, minus the canvas)
    player_count: usize,
    players: [Player; MAX_PLAYER_COUNT],
    edges: [[f32; 6]; MAP_BOX_EDGES_LEN],

    last_frame: Option<Instant>,
    rcx: Option<RenderContext>,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    pipeline_layout: Arc<PipelineLayout>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
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
            instance,
            device,
            queue,
            memory_allocator,
            command_buffer_allocator,
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
        let surface = Surface::from_window(&self.instance, &window).unwrap();
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, &Default::default())
                .unwrap();
            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, &Default::default())
                .unwrap()[0];

            Swapchain::new(
                &self.device,
                &surface,
                &SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(3),
                    image_format,
                    image_extent: window_size.into(),
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
        };

        let render_pass = single_pass_renderpass!(
            &self.device,
            attachments: {
                color: {
                    format: swapchain.image_format(),
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
        .unwrap();

        let framebuffers = window_size_dependent_setup(&images, &render_pass);

        let (pipeline, pipeline_layout) = {
            let vs = unsafe { vs::load(&self.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let fs = unsafe { fs::load(&self.device) }
                .unwrap()
                .entry_point("main")
                .unwrap();
            let vertex_input_state = LineVertex::per_vertex().definition(&vs).unwrap();
            let stages = [
                PipelineShaderStageCreateInfo::new(&vs),
                PipelineShaderStageCreateInfo::new(&fs),
            ];
            let layout = PipelineLayout::from_stages(&self.device, &stages).unwrap();
            let subpass = Subpass::new(&render_pass, 0).unwrap();

            let pipeline = GraphicsPipeline::new(
                &self.device,
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
        };

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

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

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
                if state == ElementState::Pressed {
                    if let Some(index) = self.whose_mouse(device_id) {
                        shoot(index, &mut self.players, self.player_count);
                    }
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
            &self.memory_allocator,
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
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
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
                .set_scissor(0, [region.scissor.clone()].into_iter().collect())
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
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::new(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }
}

/// Called once during initialization, then again whenever the window is resized.
fn window_size_dependent_setup(
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

fn main() -> Result<(), impl std::error::Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);

    event_loop.run_app(&mut app)
}
