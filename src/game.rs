// Game simulation: players, physics, shooting, and device-to-player assignment.
// Kept identical to the SDL version
// (https://github.com/libsdl-org/SDL/tree/main/examples/demo/02-woodeneye-008);
// only the raw u32 device IDs became winit `DeviceId`s.

use winit::event::DeviceId;

use crate::map::MAP_BOX_SCALE;

pub(crate) const MAX_PLAYER_COUNT: usize = 4; // Maximum number of players

// Mouse/keyboard rotation sensitivity, identical to the SDL version (per pixel of motion).
// This is 0x00080000 in the original C demo; the larger 0x00400000 made look 8x too fast.
pub(crate) const LOOK_SENSITIVITY: f64 = 524_288.0; // 0x00080000

// Structure representing a player.
// The SDL version stores raw u32 device IDs; winit uses its own `DeviceId` type instead.
#[derive(Clone, Copy)]
pub(crate) struct Player {
    pub(crate) mouse: Option<DeviceId>, // ID of the mouse associated with the player
    pub(crate) keyboard: Option<DeviceId>, // ID of the keyboard associated with the player
    pub(crate) pos: [f64; 3],           // 3D position of the player (x, y, z)
    pub(crate) vel: [f64; 3],           // 3D velocity of the player (x, y, z)
    pub(crate) yaw: u32,                // Horizontal rotation of the player (angle)
    pub(crate) pitch: i32,              // Vertical rotation of the player (angle)
    pub(crate) radius: f32,             // Radius of the player's collision circle
    pub(crate) height: f32,             // Height of the player
    pub(crate) color: [u8; 3],          // RGB color of the player
    pub(crate) wasd: u8, // Bitmask representing WASD key presses (Up, Left, Down, Right)
}

// Function to find a player by their mouse ID
pub(crate) fn whose_mouse(
    mouse: DeviceId,
    players: &[Player],
    _players_len: usize,
) -> Option<usize> {
    players.iter().position(|p| p.mouse == Some(mouse))
}

// Function to find a player by their keyboard ID
pub(crate) fn whose_keyboard(
    keyboard: DeviceId,
    players: &[Player],
    _players_len: usize,
) -> Option<usize> {
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
pub(crate) fn shoot(shooter: usize, players: &mut [Player], players_len: usize) {
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
    for (i, target) in players.iter_mut().enumerate().take(players_len) {
        if i == shooter {
            continue; // Skip the shooter themselves
        }
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
pub(crate) fn update(players: &mut [Player], players_len: usize, dt_ns: u64) {
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

pub(crate) fn init_players(players: &mut [Player], len: usize) {
    // Initialize player positions. Players are placed in a grid-like pattern.
    for (i, player) in players.iter_mut().enumerate().take(len) {
        player.radius = 0.5;
        player.height = 1.5;

        // Spawn halfway between the center and each wall, standing on the floor: `update`
        // clamps y to height - scale, which is exactly the standing eye height, so nobody
        // starts floating in mid-air.
        let half = MAP_BOX_SCALE as f64 * 0.5;
        player.pos[0] = half * if i & 1 != 0 { -1.0 } else { 1.0 };
        player.pos[1] = player.height as f64 - MAP_BOX_SCALE as f64;
        player.pos[2] =
            half * if i & 1 != 0 { -1.0 } else { 1.0 } * if i & 2 != 0 { -1.0 } else { 1.0 };

        player.vel[0] = 0.0;
        player.vel[1] = 0.0;
        player.vel[2] = 0.0;

        // The bitwise operations distribute the players around the origin.
        player.yaw = 0x20000000
            + if i & 1 != 0 { 0x80000000 } else { 0 }
            + if i & 2 != 0 { 0x40000000 } else { 0 };

        player.pitch = -0x08000000;

        player.wasd = 0;

        player.mouse = None;
        player.keyboard = None;

        // Generate a variety of colors per player index (unchanged from the SDL version).
        player.color[0] = if (1 << (i / 2)) & 2 != 0 { 0 } else { 0xff };
        player.color[1] = if (1 << (i / 2)) & 1 != 0 { 0 } else { 0xff };
        player.color[2] = if (1 << (i / 2)) & 4 != 0 { 0 } else { 0xff };

        player.color[0] = if i & 1 != 0 {
            player.color[0]
        } else {
            !player.color[0]
        };
        player.color[1] = if i & 1 != 0 {
            player.color[1]
        } else {
            !player.color[1]
        };
        player.color[2] = if i & 1 != 0 {
            player.color[2]
        } else {
            !player.color[2]
        };
    }
}
