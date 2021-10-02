extern crate byteorder;
extern crate lodepng;
extern crate nalgebra;
extern crate conv;
extern crate transpose;

use rayon::prelude::*;
use conv::prelude::*;
use csv;

//use serde::__private::de::borrow_cow_str;

// Naming conventions:                  https://doc.rust-lang.org/1.0.0/style/style/naming/README.html
// Inheritance (traits):                https://riptutorial.com/rust/example/22917/inheritance-with-traits
// Defining and Instantiating Structs:  https://doc.rust-lang.org/book/ch05-01-defining-structs.html
// Constructor (there are none):        https://doc.rust-lang.org/nomicon/constructors.html
//                                      https://subscription.packtpub.com/book/application_development/9781788623926/1/ch01lvl1sec16/using-the-constructor-pattern
// https://docs.rs/nav-types/0.1.0/nav_types/index.html
// https://users.rust-lang.org/t/reading-an-array-of-ints-from-a-file-solved/16530/10
// https://www.secondstate.io/articles/use-binary-data-as-function-input-and-output/
// why 16bit image does not work, grrr  https://github.com/kornelski/lodepng-rust/issues/26
// https://users.rust-lang.org/t/simultaneous-concurrent-read-and-write-into-a-buffer/56914
// https://levelup.gitconnected.com/working-with-csv-data-in-rust-7258163252f8
// https://docs.rs/csv/1.0.0-beta.5/csv/tutorial/index.html


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
    fn bearing(&self, p1:&PositionLLE, p2:&PositionLLE) -> f64;
    fn distance(&self, p1:&PositionLLE, p2:&PositionLLE) -> f64;
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

    fn bearing(&self, p1:&PositionLLE, p2:&PositionLLE) -> f64 {
        let phi1     = p1.lat.to_radians();
        let phi2     = p2.lat.to_radians();
        let d_lambda = (p2.lon - p1.lon).to_radians();
        let y = d_lambda.sin() * phi2.cos();
        let x = phi1.cos() * phi2.sin() - phi1.sin()*phi2.cos() * d_lambda.cos();
        let theta = y.atan2(x);
        return theta.to_degrees().rem_euclid(360.0);
    }

    fn distance(&self, p1:&PositionLLE, p2:&PositionLLE) -> f64 {
        // Haversine formula
        /* 
        let phi1     = p1.lat.to_radians();
        let phi2     = p2.lat.to_radians();
        let d_phi_2_sin    = ((phi2 - phi1)/2.0).sin();
        let d_lambda_2_sin = ((p2.lon - p1.lon).to_radians()/2.0).sin();
        let a = d_phi_2_sin*d_phi_2_sin  +  phi1.cos()*phi2.cos() * d_lambda_2_sin*d_lambda_2_sin;
        let c = 2.0 * a.sqrt().atan2( (1.0-a).sqrt() );
        return self.r * c;
        */
        let lat1 = p1.lat.to_radians();
        let lat2 = p2.lat.to_radians();
        let lon1 = p1.lon.to_radians();
        let lon2 = p2.lon.to_radians();
        let sin_diff_lat_2 = ((lat2-lat1)*0.5).sin();
        let sin_diff_lon_2 = ((lon2-lon1)*0.5).sin();
        let temp:f64 = sin_diff_lat_2*sin_diff_lat_2 + lat1.cos()*lat2.cos()*sin_diff_lon_2*sin_diff_lon_2; 
        return 2.0 * temp.sqrt().asin() * self.r;
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
        let data_height:usize    = tiles_vert  * 1200 + 1;
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
        let c:usize = (x as usize).min(self.data_width-1);
        let r:usize = (y as usize).min(self.data_height-1);
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

    let mut buffer: Vec<u16> = vec![0; range.array_size()];

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
            unsafe {
                let src_ptr = tile.as_ptr().offset((y*1201) as isize);
                let dst_ptr = buffer.as_mut_ptr().offset(offset as isize);
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, 1201);
            }
        }
    });
    return buffer;
    //lodepng::encode_file("test.png", buffer.as_slice(), 8401, 4801, lodepng::ColorType::LCT_GREY, 16);
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

    out_width:usize,
    out_height:usize,

    v_up:nalgebra::Vector3<f64>,
    v_north:nalgebra::Vector3<f64>,
    v_east:nalgebra::Vector3<f64>
}

impl View {
    fn new(/*earth_model:Box<dyn EarthModel>*/earth_model:Sphere, eye:PositionLLE,
            azimuth_min_r:f64, azimuth_max_r:f64, elevation_min_r:f64, elevation_max_r:f64, angular_step_r:f64,
            dist_max_m:f64, refraction_coef:f64
     ) -> Self {
         use nalgebra::{vector};

        let mut az_min_r =azimuth_min_r;
        if azimuth_min_r > azimuth_max_r {
            az_min_r -= 2.0 * std::f64::consts::PI;
        }
        // TODO: throw if > 2pi
        let azimuth_delta = azimuth_max_r - az_min_r;
        //let az_min_r = az_min_r.rem_euclid(2.0 * PI);
        //let az_max_r = az_min_r + azimuth_delta;
        // FIXME: added +1 to match Julia
        let out_width  = ((azimuth_delta/angular_step_r) as usize) + 1 + 1;
        let out_height = (((elevation_max_r - elevation_min_r)/angular_step_r) as usize) + 1 + 1;

        let eye_xyz = earth_model.lle_to_xyz(&eye);
        let ref_point = vector![eye_xyz.x, eye_xyz.y, eye_xyz.z];
        let v_z       = vector![0.0,0.0,1.0];
        let v_up      = ref_point.normalize();
        let v_down    = -v_up;
        let v_east    = v_down.cross(&v_z).normalize();
        let v_north   = v_east.cross(&v_down).normalize();
        //ref

        return View{
            earth_model, eye,
            azimuth_min_r:az_min_r, azimuth_max_r, elevation_min_r, elevation_max_r, angular_step_r,
            dist_max_m, dist_step_m:50.0,
            refraction_coef,
            out_width, out_height,
            v_up, v_north, v_east
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

fn elevation_drop_at_distance(distance:f64, radius:f64) -> f64 {
     (radius*radius-distance*distance).sqrt()-radius
}

fn precompute_earth_curve(radius: f64, dist_max: f64, dist_step:f64) -> Vec<f64> {
    let n_steps = (dist_max / dist_step) as usize + 1;
    return (0..n_steps).map(|step|{
        elevation_drop_at_distance((step as f64) * dist_step, radius)
    }).collect();
}


fn make_dist_map(view:&View, range:&LatLonRange, height_map:&Vec<u16>) -> Vec<u16> {
    use nalgebra::vector;

    let eye_xyz = view.earth_model.lle_to_xyz(&view.eye);
    let ref_point = vector![eye_xyz.x, eye_xyz.y, eye_xyz.z];
    let local_earth_radius = ref_point.norm();
    let fake_earth_radius = local_earth_radius * view.refraction_coef;
    let earth_curve = precompute_earth_curve(fake_earth_radius, view.dist_max_m, view.dist_step_m);

    println!("Earth radius is {:6.1} km (refraction x{:4.2})", local_earth_radius/1000.0, view.refraction_coef);
    println!("Output size is {} x {} pixels", view.out_width, view.out_height);
    println!("Output resolution is {} mrad per pixel or {} pixels per degree", view.angular_step_r * 1000.0, view.angular_step_r.to_degrees().recip() );
    
    let mut buffer: Vec<u16> = Vec::with_capacity(view.array_size());
    unsafe { buffer.set_len(view.array_size()); }

    let chunk_size = view.out_height;
    buffer.par_chunks_mut(chunk_size)
          .into_par_iter()
          .enumerate()
          .for_each(|(x, slice)| 
    {
        let flt_x = f64::value_from(x).expect("Conversion expect");
        let azimuth_r = view.azimuth_min_r + flt_x * view.angular_step_r;
        let mut elevation_r = view.elevation_min_r;
        let h0 = view.eye.ele;
        let n_dist_steps = (view.dist_max_m / view.dist_step_m) as usize + 1;
        let direction = view.v_north * azimuth_r.cos() + view.v_east * azimuth_r.sin();
        for i in 1..n_dist_steps {
            let dist = (i as f64) * view.dist_step_m;
            let point = ref_point + dist * direction;
            let point_lle = view.earth_model.xyz_to_lle(&PositionXYZ{x:point[0], y:point[1], z:point[2]});
            let raycast_height = h0 + elevation_r.sin() * dist;
            let terrain_height = earth_curve[i] + (height_map[range.lat_lon_to_index(point_lle.lat, point_lle.lon)] as f64);
            if terrain_height > raycast_height {
                let new_elevation_r = (terrain_height-h0).atan2(dist);
                let y_top = ((view.elevation_max_r - new_elevation_r)/view.angular_step_r) as usize;
                let y_bot = ((view.elevation_max_r - elevation_r)/view.angular_step_r) as usize;
                let v = (dist / view.dist_step_m) as u16;
                for y in y_top..=y_bot {
                    unsafe {
                        *slice.get_unchecked_mut(y) = v;
                    }
                }
                elevation_r = new_elevation_r;
            }
        }
    });

    //use std::time::Instant;
    //let timer_start = Instant::now();
    let mut output: Vec<u16> = Vec::with_capacity(view.array_size());
    unsafe { output.set_len(view.array_size()); }
    transpose::transpose(&buffer, &mut output, view.out_height, view.out_width);
    //println!("Transpose took {} seconds", timer_start.elapsed().as_secs_f64()); 11ms

    return output;
}


fn extract_outlines(view:&View, dist_map:&Vec<u16>) -> Vec<u8> {
    let mut result:Vec<u8> = Vec::with_capacity(view.array_size());
    unsafe { result.set_len(view.array_size()) }
    for col in 0..view.out_width {
        result[col] = 255;
    }
    for row in 1..view.out_height {
        for col in 0..view.out_width {
            let mut diff = (dist_map[view.out_width * (row - 1) + col] as i16) - (dist_map[view.out_width * row + col] as i16);
            diff = diff.abs().clamp(0, 255);
            result[view.out_width * row + col] = 255 - (diff as u8);
        }
    }
    return result;
}


fn test_pixel(view:&View, dist_map:&Vec<u16>, x:usize, y:usize, radius:usize, value:u16, tolerance:u16) -> bool {
    if x < (radius+1) || y < (radius+1) || (x+radius)>view.out_width || (y+radius)>view.out_height {
        return false;
    }
    for row in (y-radius)..(y+radius) {
        for col in (x-radius)..(x+radius) {
            let mapValue = dist_map[row * view.out_width + col];
            if mapValue >= value {
                if (mapValue - value) <= tolerance {
                    return true
                }
            }
            else if (value - mapValue) <= tolerance {
                return true
            }
        }
    }
    return false;
}


/******************************************************************
 _   _ _ _ _ 
| | | (_) | |
| |_| | | | |
|  _  | | | |
|_| |_|_|_|_|
******************************************************************/


struct Hill {
    name:String,
    lle:PositionLLE
}


impl Hill {    
    fn new<A>(args: A) -> Hill
    where A: IntoHill {
        args.into()
    }
}

trait IntoHill {
    fn into(self) -> Hill;
}


impl IntoHill for (std::string::String, f64, f64, f64) {
    fn into(self) -> Hill {
        assert!(self.2 <= 360.0);
        assert!(self.3 <= 360.0);
        return Hill{name:self.0.clone(), lle:PositionLLE{lat:self.2, lon:self.3, ele:self.1}};
    }
}


/******************************************************************
    _                      _        _   _
   / \   _ __  _ __   ___ | |_ __ _| |_(_) ___  _ __  ___
  / _ \ | '_ \| '_ \ / _ \| __/ _` | __| |/ _ \| '_ \/ __|
 / ___ \| | | | | | | (_) | || (_| | |_| | (_) | | | \__ \
/_/   \_\_| |_|_| |_|\___/ \__\__,_|\__|_|\___/|_| |_|___/
******************************************************************/


fn draw_annotations(view:&View, dist_map:&Vec<u16>, outlines:&Vec<u8>)
{
    use nalgebra::vector;

    let eye_xyz = view.earth_model.lle_to_xyz(&view.eye);
    let ref_point = vector![eye_xyz.x, eye_xyz.y, eye_xyz.z];
    let local_earth_radius = ref_point.norm();
    let fake_earth_radius = local_earth_radius * view.refraction_coef;

    // TODO: prepare drawing canvas
    // TODO: find font/line drawing library (https://github.com/mooman219/fontdue?)
    println!("Loading summit database");    
    let mut reader = csv::ReaderBuilder::new().delimiter(b'\t').from_path("data-cz-prom100.tsv").unwrap();
    type RecordType = (String, f64, f64, f64);
    for wrapped_record in reader.deserialize() {
        let record: RecordType = wrapped_record.unwrap();
        let hill = Hill::new(record);
        let distance_m = view.earth_model.distance(&view.eye, &hill.lle);
        if distance_m > view.dist_max_m {
            //println!("Hill {} rejected due to distance {:6.2} km", hill.name, distance_m*1.0e-3);
            continue;
        }
        let azimuth_r = view.earth_model.bearing(&view.eye, &hill.lle).to_radians();
        // FIXME: weird numbers, euclidian?
        if azimuth_r < view.azimuth_min_r || azimuth_r > view.azimuth_max_r {
            //println!("Hill {} rejected due to azimuth {:6.2}", hill.name, azimuth_r.to_degrees());
            continue;
        }
        let h = hill.lle.ele + elevation_drop_at_distance(distance_m, fake_earth_radius) - view.eye.ele;
        let elevation_r = h.atan2(distance_m); // FIXME: 
        let test_x:usize = ((azimuth_r - view.azimuth_min_r) / view.angular_step_r).round() as usize;
        let test_y:usize = ((view.elevation_max_r - elevation_r) / view.angular_step_r).round() as usize;
        let is_visible = test_pixel(view, dist_map, test_x, test_y, 4, (distance_m/view.dist_step_m) as u16, 5);
        if is_visible == false {
            //println!("Hill {} is not visible", hill.name);
            continue;
        }
        println!("{:>20} is visible at azimuth {:6.2} deg and distance {:6.2} km", hill.name, azimuth_r.to_degrees(), distance_m * 1.0e-3);
    }
    // TODO: draw azimuth ticks, horizon
}

/******************************************************************
 __  __       _
|  \/  | __ _(_)_ __
| |\/| |/ _` | | '_ \
| |  | | (_| | | | | |
|_|  |_|\__,_|_|_| |_|
******************************************************************/

fn main() {
    use std::time::Instant;

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
    let load_start = Instant::now();
    let height_map = load_data(&range);
    println!("Loading took {} seconds", load_start.elapsed().as_secs_f64());
    //println!("Height at {}, {} is {}", eye.lat, eye.lon, get_height(&range, &data, eye.lat, eye.lon));
    let view = View::new(
        spheroid, eye, 
                (90.0 as f64).to_radians(), (135.0 as f64).to_radians(), -0.0560, 0.0339, 0.0001, 
                250.0e3, 1.18);

    let dist_map_start = Instant::now();
    let dist_map = make_dist_map(&view, &range, &height_map);
    println!("Distance map took {} seconds", dist_map_start.elapsed().as_secs_f64());
    println!("Saving dist_map.png");
    lodepng::encode_file("dist_map.png", &dist_map.as_slice(), view.out_width,view.out_height, lodepng::ColorType::GREY, 16).unwrap();
    println!("Extracting outlines");
    let outlines = extract_outlines(&view, &dist_map);
    println!("Saving outlines.png");
    lodepng::encode_file("outlines.png", &outlines[..], view.out_width,view.out_height, lodepng::ColorType::GREY, 8).unwrap();
    draw_annotations(&view, &dist_map, &outlines);
    println!("All done.")
}
