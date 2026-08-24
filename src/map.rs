// Map geometry: the wireframe box the game takes place in.

// Constants defining map size. The original demo uses 16. 20 grows the arena to 125x the
// volume of the earlier 4-cube (5x longer per side), per the "more than 100 times" request.
pub(crate) const MAP_BOX_SCALE: i32 = 20;
pub(crate) const MAP_BOX_EDGES_LEN: usize = 12 + (MAP_BOX_SCALE * 2) as usize; // Number of map edges

pub(crate) fn init_edges(scale: i32, edges: &mut [[f32; 6]], _edges_len: usize) {
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
