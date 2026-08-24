// CPU-side scene building: the exact same clipping/projection math as the SDL version's
// `draw` function, but instead of painting to a canvas it emits window pixel-space line
// segments (`LineVertex`) plus the split-screen region layout for the renderer to rasterize.

use vulkano::{
    buffer::BufferContents,
    pipeline::graphics::{vertex_input::Vertex, viewport::Scissor},
};

use crate::game::Player;

const CIRCLE_DRAW_SIDES: usize = 32; // Number of sides for drawing circles

/// One vertex of a line segment, in window pixel coordinates (y down), with a color.
#[derive(BufferContents, Vertex)]
#[repr(C)]
pub(crate) struct LineVertex {
    #[format(R32G32_SFLOAT)]
    pub(crate) position: [f32; 2],
    #[format(R8G8B8A8_UNORM)]
    pub(crate) color: [u8; 4],
}

// A split-screen region: which scissors rectangle it occupies and where its vertices live.
pub(crate) struct RegionGeometry {
    pub(crate) scissor: Scissor,
    pub(crate) first_vertex: u32,
    pub(crate) vertex_count: u32,
}

// Port of the original `draw_clipped_segment`, minus the actual drawing: returns the projected
// 2D offsets from the viewport origin after clipping against the near plane z = -w.
fn project_clipped_segment(
    mut a: [f32; 3],
    mut b: [f32; 3],
    z: f32,
    w: f32,
) -> Option<([f32; 2], [f32; 2])> {
    // Both points behind the clipping plane: nothing to draw
    if a[2] >= -w && b[2] >= -w {
        return None;
    }

    let dx = a[0] - b[0];
    let dy = a[1] - b[1];

    // Clip the first point (A) if it's behind the clipping plane
    if a[2] > -w {
        let t = (-w - b[2]) / (a[2] - b[2]);
        a[0] = b[0] + dx * t;
        a[1] = b[1] + dy * t;
        a[2] = -w;
    }

    // Clip the second point (B) if it's behind the clipping plane
    if b[2] > -w {
        let t = (-w - a[2]) / (b[2] - a[2]);
        b[0] = a[0] - dx * t;
        b[1] = a[1] - dy * t;
        b[2] = -w;
    }

    // Perspective projection: project the 3D points to 2D offsets
    Some((
        [-z * a[0] / a[2], -z * a[1] / a[2]],
        [-z * b[0] / b[2], -z * b[1] / b[2]],
    ))
}

// Builds all line-segment vertices for the current frame. This mirrors the original `draw`
// function: same viewport splitting, same view matrix, same clipping/projection, same colors.
pub(crate) fn build_scene(
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

            if let Some((pa, pb)) =
                project_clipped_segment([ax, ay, az], [bx, by, bz], cam_origin, 1.0)
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
        for (j, target) in players.iter().enumerate().take(players_len) {
            if i == j {
                continue; // Don't draw the current player
            }
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
