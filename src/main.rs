//use std::fs::File;
//use std::path::Path;

// Naming conventions:                  https://doc.rust-lang.org/1.0.0/style/style/naming/README.html
// Inheritance (traits):                https://riptutorial.com/rust/example/22917/inheritance-with-traits
// Defining and Instantiating Structs:  https://doc.rust-lang.org/book/ch05-01-defining-structs.html
// Constructor (there are none):        https://doc.rust-lang.org/nomicon/constructors.html
//                                      https://subscription.packtpub.com/book/application_development/9781788623926/1/ch01lvl1sec16/using-the-constructor-pattern
pub struct PositionLLE {
    lat: f64,
    lon: f64,
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
    fn lle_to_xyz(&self, lle:PositionLLE) -> PositionXYZ {
        return PositionXYZ{x: 0.0, y:0.0, z:0.0};
    }
    fn xyz_to_lle(&self, xyz:PositionXYZ)  -> PositionLLE {
        return PositionLLE{lat: 0.0, lon:0.0, ele:0.0};
    }
}


fn main() {
    println!("Hello, world!");

    let spheroid = Sphere::new();
    //let earth_model:Box<dyn EarthModel> = Box::new(Sphere::new());
    println!("Earth diameter is {} meters", spheroid.r);
    let xyz = spheroid.lle_to_xyz(PositionLLE{lat: 49.0, lon: 16.0, ele:0.0});
    println!(" xyz = {}, {}, {}", xyz.x, xyz.y, xyz.z)


}
