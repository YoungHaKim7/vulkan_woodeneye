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
// Module layout (the code is split by function):
// - `game`    : players, physics, shooting, device assignment (identical to the SDL version),
// - `map`     : the wireframe box geometry the game takes place in,
// - `scene`   : CPU-side clipping/projection, builds the frame's line vertices + viewports,
// - `renderer`: Vulkan instance/device/swapchain/render pass/pipeline + shaders,
// - `app`     : winit event handling, input mapping, and the per-frame draw loop.
//
// Deliberate deviations from the SDL version's gameplay (see `map` and `game` for details):
// - the map box is 20 units (MAP_BOX_SCALE), 125x the volume of the 4-cube this port started
//   with, and players spawn standing on the floor instead of falling into a huge box from
//   mid-height,
// - movement acceleration scales with the box size (see `game::update`),
// - mouse look consumes winit's relative DeviceEvent::MouseMotion deltas with the original
//   demo's sensitivity, instead of diffing absolute cursor positions.

mod app;
mod game;
mod map;
mod renderer;
mod scene;

use winit::event_loop::EventLoop;

fn main() -> Result<(), impl std::error::Error> {
    let event_loop = EventLoop::new().unwrap();
    let mut app = app::App::new(&event_loop);

    event_loop.run_app(&mut app)
}
