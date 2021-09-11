//use std::path::Path;
extern crate byteorder;

// Naming conventions:                  https://doc.rust-lang.org/1.0.0/style/style/naming/README.html
// Inheritance (traits):                https://riptutorial.com/rust/example/22917/inheritance-with-traits
// Defining and Instantiating Structs:  https://doc.rust-lang.org/book/ch05-01-defining-structs.html
// Constructor (there are none):        https://doc.rust-lang.org/nomicon/constructors.html
//                                      https://subscription.packtpub.com/book/application_development/9781788623926/1/ch01lvl1sec16/using-the-constructor-pattern
// https://docs.rs/nav-types/0.1.0/nav_types/index.html


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
    f:f64,
    e:f64,
    e_sq:f64
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

fn read_tile()
{
    use std::fs::File;
    use std::io::BufReader;
    use byteorder::{BigEndian, ReadBytesExt};

    //let mut buffer = [0u16; 1201*1201];
    let array_size = 1201 * 1201;
    let file_name = "d:/_disk_d_old/devel-python/panorama/data_srtm/N49E016.hgt";
    let mut buffer: Vec<u16> = Vec::with_capacity(1201*1201);
    let file = File::open(file_name).expect("Failed to open file!");
    let mut reader = BufReader::new(file);
    unsafe { buffer.set_len(array_size); }
    reader.read_u16_into::<BigEndian>(&mut buffer[..]).expect("Failed to read file!");
    println!("buf size = {}", buffer.len());
    println!("buf[0]={}", buffer[0]);


}

fn main() {
    println!("Hello, world!");

    let spheroid = Sphere::new();
    //let earth_model:Box<dyn EarthModel> = Box::new(Sphere::new());
    println!("Earth diameter is {} meters", spheroid.r);
    let xyz = spheroid.lle_to_xyz(PositionLLE{lat: -49.0, lon: -16.0, ele:1000.0});
    println!(" xyz = {}, {}, {}", xyz.x, xyz.y, xyz.z);
    let lle = spheroid.xyz_to_lle(xyz);
    println!(" lle = {}, {}, {}", lle.lat, lle.lon, lle.ele);

    read_tile();
}
