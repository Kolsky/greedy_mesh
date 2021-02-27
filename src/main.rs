use std::cmp::Ordering;

trait AsArrayOf16<T> {
    fn as_array_of_16(self) -> [T; 16];
}

impl<T, I> AsArrayOf16<T> for I where
T: Copy + Default,
I: IntoIterator<Item = T> {
    fn as_array_of_16(self) -> [T; 16] {
        use std::mem::MaybeUninit;
        let mut result: [MaybeUninit<T>; 16] = unsafe {
            MaybeUninit::uninit().assume_init()
        };
        let default_repeater = std::iter::repeat_with(T::default);
        let yield_16 = self.into_iter()
        .chain(default_repeater)
        .enumerate()
        .take(16);
        for (i, t) in yield_16 {
            result[i] = MaybeUninit::new(t);
        }
        unsafe { result.as_ptr().cast::<[T; 16]>().read() }
    }
}

// Voxel is set whenever its bit is set.
// Data is stored as follows:
// u16 for integer xs from 0 to 15,
// index modulo 16 for integer ys from 0 to 15,
// index divided by 16 for integer zs from 0 to 15.
#[derive(Clone, Copy)]
pub struct Chunk3 {
    data: [u16; 16 * 16],
}

impl Chunk3 {
    pub fn xy_planes(self) -> [Chunk2; 16] {
        self.data
        .chunks_exact(16)
        .map(Chunk2::from)
        .as_array_of_16()
    }
}

// 2d plane with 16×16 size.
// Same as its 3-dimensional counterpart,
// except it doesn't suggest which dims are in use.
#[derive(Clone, Copy, Default)]
pub struct Chunk2 {
    data: [u16; 16],
}

impl From<&[u16]> for Chunk2 {
    fn from(slice: &[u16]) -> Self {
        Self { data: slice.iter().copied().as_array_of_16() }
    }
}

impl Chunk2 {
    pub fn greedy_mesh(self) -> Vec<MinQuad> {
        // Partially built quads. Rows contained in options allow to differentiate them.
        let pquads: &mut [Option<usize>; 16] = &mut [None; 16];
        let mut quads: Vec<MinQuad> = Vec::new();
        for (crow, &bits) in self.data.iter().enumerate() {
            let mut constructing_pquad = false;
            let mut last_quad_not_ready = false;
            let mut may_merge_length = 0;
            for bit in 0..pquads.len() {
                let cbit = bits & (1 << bit) != 0;
                if let Some(row) = pquads[bit] {
                    if cbit {
                        if may_merge_length > 0 {
                            may_merge_length += 1;
                        }
                        else if constructing_pquad || !last_quad_not_ready {
                            may_merge_length = 1;
                        }
                        else {
                            pquads[bit] = Some(crow);
                            let len = quads.len();
                            quads[len - 1].w += 1;
                        }
                    }
                    else {
                        let left_neighbour_to_pbit = 
                            bit.checked_sub(1)
                            .and_then(|i| {
                                pquads.get(i)
                                .copied()
                                .flatten() 
                            });
                        if matches!(left_neighbour_to_pbit, Some(r) if r == row) {
                            for i in (bit - may_merge_length)..bit {
                                pquads[i] = Some(row);
                            }
                        }
                        else {
                            may_merge_length = 0;
                        }
                        let len = quads.len();
                        if last_quad_not_ready && quads[len - 1].y == row as u8 {
                            quads[len - 1].w += 1;
                        }
                        else {
                            quads.push(MinQuad {
                                x: bit as u8,
                                y: row as u8,
                                w: 1 + may_merge_length as u8,
                                h: (crow - row) as u8,
                            })
                        }
                        pquads[bit] = None;
                        last_quad_not_ready = true;
                        constructing_pquad = false;
                        may_merge_length = 0;
                    }
                }
                else {
                    last_quad_not_ready = false;
                    constructing_pquad = cbit;
                    may_merge_length = 0;
                    pquads[bit] = cbit.then(|| crow);
                }
            }
        }
        let mut last_pquad = None;
        for (bit, &mut pixel) in pquads.into_iter().enumerate() {
            match (pixel, last_pquad) {
                (Some(row), Some(other_row)) if row == other_row => {
                    let len = quads.len();
                    quads[len - 1].w += 1;
                }
                (Some(row), _) => {
                    quads.push(MinQuad {
                        x: bit as u8,
                        y: row as u8,
                        w: 1,
                        h: (16 - row) as u8,
                    })
                }
                _ => {}
            }
            last_pquad = pixel;
        }
        quads
    }
}

// The naming for fields and the struct itself is a bit confusing here.
// MinQuad represents an integer rectangle at Chunk2.
// MinQuads have a total ordering, which allows to build
// a greedy mesh for 2d plane.
// For any (row, bit) combination there is a pixel
// if x <= bit <= (x + w) and y <= row <= (y + h).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, Hash)]
pub struct MinQuad {
    // dimension 0 (bit) min coord
    pub x: u8,
    // dimension 1 (row) min coord
    pub y: u8,
    // dimension 0 max coord - min coord
    pub w: u8,
    // dimension 1 max coord - min coord
    pub h: u8,
}

impl PartialOrd for MinQuad {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.y != other.y { return self.y.partial_cmp(&other.y); }
        if self.x != other.x { return self.x.partial_cmp(&other.x); }
        if self.w != other.w { return other.w.partial_cmp(&self.w); }
        other.h.partial_cmp(&self.h)
    }
}

fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use super::*;

    #[test]
    fn greedy_mesh_is_correct_for_plane() {
        let data: [u16; 16] = [
            0b_0000_0000_0000_0000_u16.reverse_bits(),
            0b_0000_0000_0000_0000_u16.reverse_bits(),
            0b_1110_0000_0000_0000_u16.reverse_bits(),
            0b_1111_1000_0000_0000_u16.reverse_bits(),
            0b_0100_1000_0000_0000_u16.reverse_bits(),
            0b_0000_0111_1010_0000_u16.reverse_bits(),
            0b_0000_0011_0011_1000_u16.reverse_bits(),
            0b_0000_0001_1110_0000_u16.reverse_bits(),
            0b_0000_0000_0000_0000_u16.reverse_bits(),
            0b_0000_0001_0000_0001_u16.reverse_bits(),
            0b_0000_0011_1000_0001_u16.reverse_bits(),
            0b_0000_0111_1100_0001_u16.reverse_bits(),
            0b_0000_1111_1110_0001_u16.reverse_bits(),
            0b_0001_1111_1111_0001_u16.reverse_bits(),
            0b_0011_1111_1111_1000_u16.reverse_bits(),
            0b_0111_1111_1111_1100_u16.reverse_bits(),
        ];
        let chunk2 = Chunk2 { data };
        let quads: HashSet<_> = chunk2.greedy_mesh().into_iter().collect();
        assert_eq!(quads, vec![
            MinQuad { x: 0, y: 2, w: 3, h: 2 },
            MinQuad { x: 3, y: 3, w: 2, h: 1 },
            MinQuad { x: 1, y: 4, w: 1, h: 1 },
            MinQuad { x: 4, y: 4, w: 1, h: 1 },
            MinQuad { x: 5, y: 5, w: 4, h: 1 },
            MinQuad { x: 6, y: 6, w: 2, h: 1 },
            MinQuad { x: 11, y: 6, w: 2, h: 1 },
            MinQuad { x: 7, y: 7, w: 3, h: 1 },
            MinQuad { x: 10, y: 5, w: 1, h: 3 },
            MinQuad { x: 15, y: 9, w: 1, h: 5 },
            MinQuad { x: 1, y: 15, w: 1, h: 1 },
            MinQuad { x: 2, y: 14, w: 1, h: 2 },
            MinQuad { x: 3, y: 13, w: 1, h: 3 },
            MinQuad { x: 4, y: 12, w: 1, h: 4 },
            MinQuad { x: 5, y: 11, w: 1, h: 5 },
            MinQuad { x: 6, y: 10, w: 1, h: 6 },
            MinQuad { x: 7, y: 9, w: 1, h: 7 },
            MinQuad { x: 8, y: 10, w: 1, h: 6 },
            MinQuad { x: 9, y: 11, w: 1, h: 5 },
            MinQuad { x: 10, y: 12, w: 1, h: 4 },
            MinQuad { x: 11, y: 13, w: 1, h: 3 },
            MinQuad { x: 12, y: 14, w: 1, h: 2 },
            MinQuad { x: 13, y: 15, w: 1, h: 1 },
        ].into_iter().collect());
    }
}