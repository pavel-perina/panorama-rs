extern crate byteorder;
extern crate lodepng;
extern crate ndarray; // lazy to do vectors

//use rayon::prelude::*;

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
    fn lle_to_xyz(&self, lle:&PositionLLE) -> PositionXYZ;
    fn xyz_to_lle(&self, xyz:&PositionXYZ) -> PositionLLE;
}


struct Sphere {
    r: f64
}


impl Sphere {
    fn new() -> Self {
        return Sphere {r:6378137.0};
    }
}

/*
struct Spheroid {
    a: f64,
    b: f64,
    f: f64,
    e: f64,
    e_sq: f64
}
*/

impl EarthModel for Sphere {

    fn lle_to_xyz(&self, p:&PositionLLE) -> PositionXYZ {
        let phi    = p.lat.to_radians();
        let lambda = p.lon.to_radians();
        let v = self.r + p.ele;
        let x = v * phi.cos() * lambda.cos();
        let y = v * phi.cos() * lambda.sin();
        let z = v * phi.sin();
        return PositionXYZ{x, y, z};
    }

    fn xyz_to_lle(&self, p:&PositionXYZ) -> PositionLLE {
        let v= (p.x*p.x + p.y*p.y + p.z*p.z).sqrt();
        let ele = v - self.r;
        let lon = p.y.atan2(p.x).to_degrees(); // WTF? atan(y/x)!
        let lat = (p.z/v).asin().to_degrees();
        return PositionLLE{lat, lon, ele};
    }
}


impl Sphere {
    fn bearing(&self, p1:&PositionLLE, p2:&PositionLLE) -> f64 {
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
    max_lon:i32,
    tiles_horiz:usize,
    tiles_vert:usize,
    data_width:usize,
    data_height:usize,
    pixels_per_deg:f64
}

impl LatLonRange {
    fn new(min_lat:i32, min_lon:i32, max_lat:i32, max_lon:i32) -> Self {
        let tiles_horiz:usize    = (max_lon - min_lon + 1) as usize;
        let tiles_vert:usize     = (max_lat - min_lat + 1) as usize;
        let pixels_per_deg:f64   = 1200.0;
        let data_width:usize     = tiles_horiz * 1200 + 1;
        let data_height:usize    = tiles_vert * 1200 + 1;
        return LatLonRange{min_lat, min_lon, max_lat, max_lon, 
                           tiles_horiz, tiles_vert, 
                           data_width, data_height,
                           pixels_per_deg};
    }
    fn tiles_total(&self) -> usize {
        self.tiles_horiz * self.tiles_vert
    }
    fn lat_lon_to_index(&self, lat:f64, lon:f64) -> usize {
        let y:f64 = ((((self.max_lat+1) as f64) - lat) * self.pixels_per_deg).max(0.0);
        let x:f64 = ((lon-(self.min_lon as f64)) * self.pixels_per_deg).max(0.0);
        let c:usize = (x as usize).min(self.data_width);
        let r:usize = (y as usize).min(self.data_height);
        //println!("x={} ({}), y={} ({}), offset={}", x,c,y,r,r*self.data_width +c);
        return r * self.data_width + c;
    }
    fn array_size(&self) -> usize {
        self.data_height * self.data_width
    }
}


fn load_data(range:&LatLonRange) -> Vec<u16> 
{
    println!("Requesting data for area {}°N {}°E - {}°N {}°E ... (aprox. {:3.0}x{:3.0} km).",
        range.min_lat, range.min_lon,
        range.max_lat, range.max_lon,
        (range.tiles_horiz as f64) * 111.1 * (range.min_lat as f64).to_radians().cos(),
        (range.tiles_vert  as f64) * 111.1
        );

    println!("I will read {}x{}={} tiles, heightmap size is {}x{} ({} MB).",
        range.tiles_horiz, range.tiles_vert, range.tiles_total(),
        range.data_width,  range.data_height, range.array_size()*2/1000000
        );

    let mut buffer: Vec<u16> = Vec::with_capacity(range.array_size());
    unsafe { buffer.set_len(range.array_size()); }

    let mut progress = 0;
    // TODO: make this parallel with mutex around print progress?
    (0..range.tiles_total()).into_iter().for_each(|i| {
        let lat = range.min_lat + ((i / range.tiles_horiz) as i32);
        let lon = range.min_lon + ((i % range.tiles_horiz) as i32);
        // Print progress
        progress += 1;
        println!("Loading tile {:03}/{:03} lat={:02}, lon={:02}", progress, range.tiles_total(), lat, lon);
        // Load tile
        let tile = load_tile(lat, lon);
        // Move to data
        let tile_y = (range.max_lat - lat) as usize;
        let tile_x = (lon - range.min_lon) as usize;
        let data_width = range.data_width;
        for y in 0..=1200 {
            let row = tile_y * 1200 + y;
            let mut offset = (row * data_width) as usize;
            offset += tile_x * 1200;
            for x in 0..=1200 {
                buffer[offset + x] = tile[y * 1201 + x];
            }
        }
    });
    return buffer;
    //lodepng::encode_file("test.png", buffer.as_slice(), 8401, 4801, lodepng::ColorType::LCT_GREY, 16);
}


fn get_height(range:&LatLonRange, data:&Vec<u16>, lat:f64, lon:f64) -> u16
{
    data[range.lat_lon_to_index(lat, lon)]
}

/******************************************************************
__     ___
\ \   / (_) _____      __
 \ \ / /| |/ _ \ \ /\ / /
  \ V / | |  __/\ V  V /
   \_/  |_|\___| \_/\_/
******************************************************************/

struct View {
    earth_model:Sphere,
    //earth_model:Box<dyn EarthModel>,
    eye:PositionLLE,

    azimuth_min_r:f64,
    azimuth_max_r:f64,
    elevation_min_r:f64,
    elevation_max_r:f64,
    angular_step_r:f64,

    dist_max_m:f64,
    dist_step_m:f64,

    refraction_coef:f64,
/*
    vUp:Vector{Float64}
    vNorth:Vector{Float64}
    vEast:Vector{Float64}
*/
    out_width:usize,
    out_height:usize,
}

impl View {
    fn new(/*earth_model:Box<dyn EarthModel>*/earth_model:Sphere, eye:PositionLLE,
            azimuth_min_r:f64, azimuth_max_r:f64, elevation_min_r:f64, elevation_max_r:f64, angular_step_r:f64,
            dist_max_m:f64, refraction_coef:f64
     ) -> Self {
        let mut az_min_r =azimuth_min_r;
        if azimuth_min_r > azimuth_max_r {
            az_min_r -= 2.0 * std::f64::consts::PI;
        }
        // TODO: throw if > 2pi
        let azimuth_delta = azimuth_max_r - az_min_r;
        //let az_min_r = az_min_r.rem_euclid(2.0 * PI);
        //let az_max_r = az_min_r + azimuth_delta;
        let out_width  = ((azimuth_delta/angular_step_r) as usize) + 1;
        let out_height = (((elevation_max_r - elevation_min_r)/angular_step_r) as usize) + 1;

        return View{
            earth_model, eye,
            azimuth_min_r:az_min_r, azimuth_max_r, elevation_min_r, elevation_max_r, angular_step_r,
            dist_max_m, dist_step_m:50.0,
            refraction_coef,
            out_width, out_height
        };
    }

    fn array_size(&self) -> usize {
        self.out_width * self.out_height
    }
}

/******************************************************************
 ____  _     _                         __  __
|  _ \(_)___| |_ __ _ _ __   ___ ___  |  \/  | __ _ _ __  
| | | | / __| __/ _` | '_ \ / __/ _ \ | |\/| |/ _` | '_ \ 
| |_| | \__ \ || (_| | | | | (_|  __/ | |  | | (_| | |_) |
|____/|_|___/\__\__,_|_| |_|\___\___| |_|  |_|\__,_| .__/ 
                                                   |_| 
******************************************************************/

fn make_dist_map(view:&View, range:&LatLonRange, height_map:&Vec<u16>) -> Vec<u16> {
    use ndarray::arr1;

    let eye_xyz = view.earth_model.lle_to_xyz(&view.eye);
    let ref_point = arr1(&[eye_xyz.x, eye_xyz.y, eye_xyz.y]);
    let local_earth_radius = ref_point.dot(&ref_point).sqrt();
    let fake_earth_radius = local_earth_radius * view.refraction_coef;

    println!("Earth radius is {:6.1} km (refraction x{:4.2})", local_earth_radius/1000.0, view.refraction_coef);
    println!("Output size is {} x {} pixels", view.out_width, view.out_height);
    println!("Output resolution is {} mrad per pixel or {} pixels per degree", view.angular_step_r * 1000.0, view.angular_step_r.to_degrees().recip() );
    
    let mut buffer: Vec<u16> = Vec::with_capacity(view.array_size());
    unsafe { buffer.set_len(view.array_size()); }



    return buffer;
}

/******************************************************************
 __  __       _
|  \/  | __ _(_)_ __
| |\/| |/ _` | | '_ \
| |  | | (_| | | | | |
|_|  |_|\__,_|_|_| |_|
******************************************************************/

fn main() {
    let range = LatLonRange::new(47,15, 50, 21);
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
    let height_map = load_data(&range);
    //println!("Height at {}, {} is {}", eye.lat, eye.lon, get_height(&range, &data, eye.lat, eye.lon));
    let view = View::new(
        spheroid, eye, 
                (90.0 as f64).to_radians(), (135.0 as f64).to_radians(), -0.0560, 0.0339, 0.0001, 
                250.0e3, 1.18);

    let dist_map = make_dist_map(&view, &range, &height_map);
}
