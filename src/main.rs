extern crate byteorder;
extern crate lodepng;
// Naming conventions:                  https://doc.rust-lang.org/1.0.0/style/style/naming/README.html
// Inheritance (traits):                https://riptutorial.com/rust/example/22917/inheritance-with-traits
// Defining and Instantiating Structs:  https://doc.rust-lang.org/book/ch05-01-defining-structs.html
// Constructor (there are none):        https://doc.rust-lang.org/nomicon/constructors.html
//                                      https://subscription.packtpub.com/book/application_development/9781788623926/1/ch01lvl1sec16/using-the-constructor-pattern
// https://docs.rs/nav-types/0.1.0/nav_types/index.html
// https://users.rust-lang.org/t/reading-an-array-of-ints-from-a-file-solved/16530/10
// https://www.secondstate.io/articles/use-binary-data-as-function-input-and-output/
// why 16bit image does not work, grrr  https://github.com/kornelski/lodepng-rust/issues/26


/******************************************************************
  ____            _   _ _   _ _
 / ___| ___  ___ | | | | |_(_) |___
| |  _ / _ \/ _ \| | | | __| | / __|
| |_| |  __/ (_) | |_| | |_| | \__ \
 \____|\___|\___/ \___/ \__|_|_|___/
******************************************************************/

pub struct PositionLLE {
    lat: f64, // latitude  | phi   | +north/-south of equator
    lon: f64, // longitude | lamda | +east/-west of Greenwich
    ele: f64
}


pub struct PositionXYZ {
    x: f64,
    y: f64,
    z: f64
}


trait EarthModel {
    fn lle_to_xyz(&self, lle:PositionLLE) -> PositionXYZ;
    fn xyz_to_lle(&self, xyz:PositionXYZ) -> PositionLLE;
}


struct Sphere {
    r: f64
}


impl Sphere {
    fn new() -> Self {
        return Sphere {r:6378137.0};
    }
}


struct Spheroid {
    a: f64,
    b: f64,
    f: f64,
    e: f64,
    e_sq: f64
}


impl EarthModel for Sphere {

    fn lle_to_xyz(&self, p:PositionLLE) -> PositionXYZ {
        let phi    = p.lat.to_radians();
        let lambda = p.lon.to_radians();
        let v = self.r + p.ele;
        let x = v * phi.cos() * lambda.cos();
        let y = v * phi.cos() * lambda.sin();
        let z = v * phi.sin();
        return PositionXYZ{x, y, z};
    }

    fn xyz_to_lle(&self, p:PositionXYZ) -> PositionLLE {
        let v= (p.x*p.x + p.y*p.y + p.z*p.z).sqrt();
        let ele = v - self.r;
        let lon = p.y.atan2(p.x).to_degrees(); // WTF? atan(y/x)!
        let lat = (p.z/v).asin().to_degrees();
        return PositionLLE{lat, lon, ele};
    }
}


impl Sphere {
    fn bearing(&self, p1:PositionLLE, p2:PositionLLE) -> f64 {
        let phi1     = p1.lat.to_radians();
        let phi2     = p2.lat.to_radians();
        let d_lambda = (p2.lon - p1.lon).to_radians();
        let y = d_lambda.sin() * phi2.cos();
        let x = phi1.cos() * phi2.sin() - phi1.sin()*phi2.cos() * d_lambda.cos();
        let theta = y.atan2(x);
        return theta.to_degrees().rem_euclid(360.0);
    }
}

/******************************************************************
 _   _      _       _     _   __  __
| | | | ___(_) __ _| |__ | |_|  \/  | __ _ _ __
| |_| |/ _ \ |/ _` | '_ \| __| |\/| |/ _` | '_ \
|  _  |  __/ | (_| | | | | |_| |  | | (_| | |_) |
|_| |_|\___|_|\__, |_| |_|\__|_|  |_|\__,_| .__/
              |___/                       |_|
******************************************************************/

fn load_tile(lat:i32, lon:i32) ->  Vec<u16>
{
    use std::fs::File;
    use std::io::BufReader;
    use byteorder::{BigEndian, ReadBytesExt};
    let array_size = 1201 * 1201;
    let file_name = format!("d:/_disk_d_old/devel-python/panorama/data_srtm/N{:02}E{:03}.hgt", lat, lon);
    let mut buffer: Vec<u16> = Vec::with_capacity(1201*1201);
    let file = File::open(file_name).expect("Failed to open file!");
    let mut reader = BufReader::new(file);
    unsafe { buffer.set_len(array_size); }
    // TODO: clamp data, convert whatever is greater than for example 10K to zero (input data are signed)
    reader.read_u16_into::<BigEndian>(&mut buffer[..]).expect("Failed to read file!");
    return buffer;
}


struct LatLonRange {
    min_lat:i32,
    min_lon:i32,
    max_lat:i32,
    max_lon:i32
}


fn load_data(range:LatLonRange) -> Vec<u16> {
    let n_tiles_horiz: usize = (range.max_lon - range.min_lon + 1) as usize;
    let n_tiles_vert: usize  = (range.max_lat - range.min_lat + 1) as usize;
    let n_tiles_total  = n_tiles_horiz * n_tiles_vert;
    let data_width     = n_tiles_horiz * 1200 + 1;
    let data_height    = n_tiles_vert  * 1200 + 1;
    let array_size     = data_width * data_height;

    println!("Requesting data for area {}°N {}°E - {}°N {}°E ... (aprox. {:3.0}x{:3.0} km).",
        range.min_lat, range.min_lon,
        range.max_lat, range.max_lon,
        (n_tiles_horiz as f64) * 111.1 * (range.min_lat as f64).to_radians().cos(),
        (n_tiles_vert  as f64) * 111.1
        );

    println!("I will read {}x{}={} tiles, heightmap size is {}x{} ({} MB).",
        n_tiles_horiz, n_tiles_vert, n_tiles_total,
        data_width, data_height, array_size*2/1000000
        );

    let mut buffer: Vec<u16> = Vec::with_capacity(array_size);
    unsafe { buffer.set_len(array_size); }

    let mut progress = 0;
    for i in 0..n_tiles_total {
        let lat = range.min_lat + ((i / n_tiles_horiz) as i32);
        let lon = range.min_lon + ((i % n_tiles_horiz) as i32);
        // Print progress
        progress += 1;
        println!("Loading tile {:03}/{:03} lat={:02}, lon={:02}", progress, n_tiles_total, lat, lon);
        // Load tile
        let tile = load_tile(lat, lon);
        // Move to data
        let tile_y = (range.max_lat - lat) as usize;
        let tile_x = (lon - range.min_lon) as usize;
        for y in 0..=1200 {
            let row = tile_y * 1200 + y;
            let mut offset = (row * data_width) as usize;
            offset += tile_x * 1200;
            for x in 0..=1200 {
                buffer[offset + x] = tile[y * 1201 + x];
            }
        }
    }
    return buffer;
    //lodepng::encode_file("test.png", buffer.as_slice(), 8401, 4801, lodepng::ColorType::LCT_GREY, 16);
}

/******************************************************************
__  __       _
|  \/  | __ _(_)_ __
| |\/| |/ _` | | '_ \
| |  | | (_| | | | | |
|_|  |_|\__,_|_|_| |_|
******************************************************************/

fn main() {
    let range = LatLonRange{min_lat: 47, min_lon: 15, max_lat: 50, max_lon: 21};
    let eye   = PositionLLE{lat: 50.08309, lon: 17.23094, ele:1510.0};
    let spheroid  = Sphere::new();    
    //let earth_model:Box<dyn EarthModel> = Box::new(Sphere::new());
    /*
    println!("Earth diameter is {} meters", spheroid.r);
    let xyz = spheroid.lle_to_xyz(PositionLLE{lat: -49.0, lon: -16.0, ele:1000.0});
    println!(" xyz = {}, {}, {}", xyz.x, xyz.y, xyz.z);
    let lle = spheroid.xyz_to_lle(xyz);
    println!(" lle = {}, {}, {}", lle.lat, lle.lon, lle.ele);
    */
    //read_tile(49, 16);
    load_data(range);

}
