use std::ops::{Add, Sub};
use std::cmp::Ordering;
use std::convert::TryInto;

trait AsArrayOfChunk2Len<T> {
    fn as_array_of_chunk2_len(self) -> [T; Chunk2::LEN];
}

impl<T, I> AsArrayOfChunk2Len<T> for I where
T: Default,
I: IntoIterator<Item = T> {
    fn as_array_of_chunk2_len(self) -> [T; Chunk2::LEN] {
        use std::mem::MaybeUninit;
        let mut result: [MaybeUninit<T>; Chunk2::LEN] = unsafe {
            MaybeUninit::uninit().assume_init()
        };
        let default_repeater = std::iter::repeat_with(T::default);
        let yield_16 = self.into_iter()
        .chain(default_repeater)
        .enumerate()
        .take(Chunk2::LEN);
        for (i, t) in yield_16 {
            result[i] = MaybeUninit::new(t);
        }
        unsafe { result.as_ptr().cast::<[T; Chunk2::LEN]>().read() }
    }
}

// Voxel is set whenever its bit is set.
// Data is stored as follows:
// u16 for integer xs from 0 to 15,
// index modulo 16 for integer ys from 0 to 15,
// index divided by 16 for integer zs from 0 to 15.
#[derive(Clone, Copy)]
pub struct Chunk3 {
    data: [u16; Self::LEN],
}

impl Chunk3 {
    const LEN: usize = Chunk2::LEN * Chunk2::LEN;

    pub fn xy_planes(self) -> [Chunk2; Chunk2::LEN] {
        self.data
        .chunks_exact(Chunk2::LEN)
        .map(Chunk2::from)
        .as_array_of_chunk2_len()
    }

    pub fn xz_planes(self) -> [Chunk2; Chunk2::LEN] {
        let mut data_vec = Vec::with_capacity(Self::LEN);
        for i in 0..Self::LEN {
            let i = i / Chunk2::LEN + Chunk2::LEN * (i % Chunk2::LEN);
            data_vec.push(self.data[i]);
        }
        let data = data_vec.as_slice().try_into().unwrap();
        Self { data }.xy_planes()
    }

    pub fn yz_planes(self) -> [Chunk2; Chunk2::LEN] {
        let mut old_data = self.data;
        let mut data = [0; Self::LEN];
        let bit_length = 8 * std::mem::size_of_val(&data[0]);
        for i in 0..Self::LEN {
            let offset =  Chunk2::LEN * (i % Chunk2::LEN);
            for bit in 0..bit_length {
                data[i] <<= 1;
                data[i] |= old_data[bit + offset] & 1;
                old_data[bit + offset] >>= 1;
            }
        }
        Self { data }.xy_planes()
    }

    pub fn greedy_mesh(self) -> Vec<Quad> {
        fn get_min_quads(planes: &[Chunk2]) -> [Vec<MinQuad>; 16] {
            let len = planes.len();
            (0..len).map(move |layer| {
                let chunk_slice: Chunk2 =
                    if layer + 1 != len {
                        planes[layer] - planes[layer + 1]
                    }
                    else {
                        planes[layer]
                    };
                chunk_slice.greedy_mesh()
            })
            .as_array_of_chunk2_len()
        };
        fn get_quads(quads: &mut Vec<Quad>, planes: &[Chunk2], min_quad_mapper: impl Fn(&MinQuad, u8) -> Quad + Copy) {
            let min_quads = get_min_quads(planes);
            let iter = 
                min_quads.iter()
                .enumerate()
                .flat_map(|(dim, v)| {
                    v.iter().map(move |mq| min_quad_mapper(mq, dim as u8))
                });
            quads.extend(iter);
        };
        let mut quads: Vec<Quad> = Vec::new();
        let inv = |dim| Chunk2::LEN as u8 - dim;

        let mut xy_planes = self.xy_planes();
        get_quads(&mut quads, &xy_planes, |mq, z| {
            Quad::new_xy(mq.dim_0, mq.dim_1, mq.dim_0 + mq.dim_0a, mq.dim_1 + mq.dim_1a, z)
        });
        xy_planes.reverse();
        get_quads(&mut quads, &xy_planes, |mq, inv_z| {
            Quad::new_xy(mq.dim_0 + mq.dim_0a, mq.dim_1, mq.dim_0, mq.dim_1 + mq.dim_1a, inv(inv_z))
        });

        let mut xz_planes = self.xz_planes();
        get_quads(&mut quads, &xz_planes, |mq, y| {
            Quad::new_xz(mq.dim_0, mq.dim_1, mq.dim_0 + mq.dim_0a, mq.dim_1 + mq.dim_1a, y)
        });
        xz_planes.reverse();
        get_quads(&mut quads, &xz_planes, |mq, inv_y| {
            Quad::new_xz(mq.dim_0 + mq.dim_0a, mq.dim_1, mq.dim_0, mq.dim_1 + mq.dim_1a, inv(inv_y))
        });

        let mut yz_planes = self.yz_planes();
        get_quads(&mut quads, &yz_planes, |mq, x| {
            Quad::new_yz(mq.dim_0, mq.dim_1, mq.dim_0 + mq.dim_0a, mq.dim_1 + mq.dim_1a, x)
        });
        yz_planes.reverse();
        get_quads(&mut quads, &yz_planes, |mq, inv_x| {
            Quad::new_yz(mq.dim_0 + mq.dim_0a, mq.dim_1, mq.dim_0, mq.dim_1 + mq.dim_1a, inv(inv_x))
        });
        
        quads
    }
}

// 2d plane with 16×16 size.
// Same as its 3-dimensional counterpart,
// except it doesn't suggest which dims are in use.
#[derive(Clone, Copy, Default)]
pub struct Chunk2 {
    data: [u16; Self::LEN],
}

impl From<&[u16]> for Chunk2 {
    fn from(slice: &[u16]) -> Self {
        Self { data: slice.iter().copied().as_array_of_chunk2_len() }
    }
}

impl Sub for Chunk2 {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        for (left, right) in &mut self.data.iter_mut().zip(&rhs.data) {
            *left &= !right;
        }
        self
    }
}

impl Chunk2 {
    const LEN: usize = 16;
    pub fn greedy_mesh(self) -> Vec<MinQuad> {
        // Partially built quads. Rows contained in options allow to differentiate them.
        let pquads: &mut [Option<usize>; Self::LEN] = &mut [None; Self::LEN];
        let mut quads: Vec<MinQuad> = Vec::new();
        for (crow, &bits) in self.data.iter().enumerate() {
            let mut constructing_pquad = false;
            let mut last_quad_not_ready = false;
            let mut may_merge_length = 0;
            let mut pquads_of_previous_bit = None;
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
                            quads[len - 1].dim_0a += 1;
                        }
                    }
                    else {
                        if pquads_of_previous_bit == pquads[bit] {
                            for i in (bit - may_merge_length)..bit {
                                pquads[i] = Some(row);
                            }
                        }
                        else {
                            may_merge_length = 0;
                        }
                        let len = quads.len();
                        if last_quad_not_ready && quads[len - 1].dim_1 == row as u8 {
                            quads[len - 1].dim_0a += 1;
                        }
                        else {
                            quads.push(MinQuad {
                                dim_0: bit as u8,
                                dim_1: row as u8,
                                dim_0a: 1 + may_merge_length as u8,
                                dim_1a: (crow - row) as u8,
                            })
                        }
                        pquads[bit] = None;
                        last_quad_not_ready = true;
                        constructing_pquad = false;
                        may_merge_length = 0;
                    }
                    pquads_of_previous_bit = pquads[bit];
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
                    quads[len - 1].dim_0a += 1;
                }
                (Some(row), _) => {
                    quads.push(MinQuad {
                        dim_0: bit as u8,
                        dim_1: row as u8,
                        dim_0a: 1,
                        dim_1a: (Self::LEN - row) as u8,
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
// if dim_0 <= bit <= (dim_0 + dim_0a) and dim_1 <= row <= (dim_1 + dim_1a).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, Hash)]
pub struct MinQuad {
    // dimension 0 (bit) min coord
    pub dim_0: u8,
    // dimension 1 (row) min coord
    pub dim_1: u8,
    // dimension 0 max coord - min coord
    pub dim_0a: u8,
    // dimension 1 max coord - min coord
    pub dim_1a: u8,
}

impl PartialOrd for MinQuad {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.dim_1 != other.dim_1 { return self.dim_1.partial_cmp(&other.dim_1); }
        if self.dim_0 != other.dim_0 { return self.dim_0.partial_cmp(&other.dim_0); }
        if self.dim_0a != other.dim_0a { return other.dim_0a.partial_cmp(&self.dim_0a); }
        other.dim_1a.partial_cmp(&self.dim_1a)
    }
}

// sf::Vector3<T> analog for demonstration purposes
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Add for Vec3<T> where T: Add<Output = T> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
        self.z = self.z + rhs.z;
        self
    }
}

impl<T> Vec3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

pub struct Quad {
    pub verts: [Vec3<u8>; 6],
}

impl Quad {
    fn from_4_vertices(vertices: [Vec3<u8>; 4]) -> Self {
        Self {
            verts: [
                vertices[0], vertices[1], vertices[2],
                vertices[3], vertices[2], vertices[1],
            ],
        }
    }

    pub fn new_xy(x0: u8, y0: u8, x1: u8, y1: u8, z: u8) -> Self {
        let vertices = [
            Vec3::new(x0, y0, z),
            Vec3::new(x0, y1, z),
            Vec3::new(x1, y1, z),
            Vec3::new(x1, y0, z),
        ];
        Self::from_4_vertices(vertices)
    }
    
    pub fn new_xz(x0: u8, z0: u8, x1: u8, z1: u8, y: u8) -> Self {
        let vertices = [
            Vec3::new(x0, y, z0),
            Vec3::new(x0, y, z1),
            Vec3::new(x1, y, z1),
            Vec3::new(x1, y, z0),
        ];
        Self::from_4_vertices(vertices)
    }
    
    pub fn new_yz(y0: u8, y1: u8, z0: u8, z1: u8, x: u8) -> Self {
        let vertices = [
            Vec3::new(x, y0, z0),
            Vec3::new(x, y0, z1),
            Vec3::new(x, y1, z1),
            Vec3::new(x, y1, z0),
        ];
        Self::from_4_vertices(vertices)
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
            MinQuad { dim_0: 0,  dim_1: 2,  dim_0a: 3, dim_1a: 2 },
            MinQuad { dim_0: 3,  dim_1: 3,  dim_0a: 2, dim_1a: 1 },
            MinQuad { dim_0: 1,  dim_1: 4,  dim_0a: 1, dim_1a: 1 },
            MinQuad { dim_0: 4,  dim_1: 4,  dim_0a: 1, dim_1a: 1 },
            MinQuad { dim_0: 5,  dim_1: 5,  dim_0a: 4, dim_1a: 1 },
            MinQuad { dim_0: 6,  dim_1: 6,  dim_0a: 2, dim_1a: 1 },
            MinQuad { dim_0: 11, dim_1: 6,  dim_0a: 2, dim_1a: 1 },
            MinQuad { dim_0: 7,  dim_1: 7,  dim_0a: 3, dim_1a: 1 },
            MinQuad { dim_0: 10, dim_1: 5,  dim_0a: 1, dim_1a: 3 },
            MinQuad { dim_0: 15, dim_1: 9,  dim_0a: 1, dim_1a: 5 },
            MinQuad { dim_0: 1,  dim_1: 15, dim_0a: 1, dim_1a: 1 },
            MinQuad { dim_0: 2,  dim_1: 14, dim_0a: 1, dim_1a: 2 },
            MinQuad { dim_0: 3,  dim_1: 13, dim_0a: 1, dim_1a: 3 },
            MinQuad { dim_0: 4,  dim_1: 12, dim_0a: 1, dim_1a: 4 },
            MinQuad { dim_0: 5,  dim_1: 11, dim_0a: 1, dim_1a: 5 },
            MinQuad { dim_0: 6,  dim_1: 10, dim_0a: 1, dim_1a: 6 },
            MinQuad { dim_0: 7,  dim_1: 9,  dim_0a: 1, dim_1a: 7 },
            MinQuad { dim_0: 8,  dim_1: 10, dim_0a: 1, dim_1a: 6 },
            MinQuad { dim_0: 9,  dim_1: 11, dim_0a: 1, dim_1a: 5 },
            MinQuad { dim_0: 10, dim_1: 12, dim_0a: 1, dim_1a: 4 },
            MinQuad { dim_0: 11, dim_1: 13, dim_0a: 1, dim_1a: 3 },
            MinQuad { dim_0: 12, dim_1: 14, dim_0a: 1, dim_1a: 2 },
            MinQuad { dim_0: 13, dim_1: 15, dim_0a: 1, dim_1a: 1 },
        ].into_iter().collect());
    }
}