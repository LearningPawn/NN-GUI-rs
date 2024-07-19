use ndarray::{Array, Dim};
use image::{ImageBuffer, GrayImage};
use std::fs::File;
use std::io::prelude::*;
use csv::{ReaderBuilder, StringRecord};
use std::error::Error;

fn get_img_pixels_data(record: &StringRecord) -> Vec<u8> {
    let mut image_vec: Vec<u8> = vec![];
    let mut count = 0;
    for data_value in record.iter() {
        if count == 0 {
            count += 1;
            continue;
        } else {
            image_vec.push(data_value.parse::<u8>().unwrap());
        }
    }
    image_vec
}

fn save_record_to_image(record: &StringRecord, file_name: &str) {
    // read digital pixels and convert to ndarray image
    let image_vec: Vec<u8> = get_img_pixels_data(record);

    let image_array: Array<u8, Dim<[usize; 2]>> = Array::from_shape_vec((28, 28), image_vec).unwrap();
    let image_buffer = ImageBuffer::from_fn(28, 28, |x, y| {
        // use [[y, x]] instead of [[x, y]] to access the pixel value at (x, y) in the pixels array is 
        // because ndarray uses row-major order to store its elements.
        let gray_value = image_array[[y as usize, x as usize]];
        GrayImage::from_pixel(1, 1, image::Luma([gray_value]))
            .get_pixel(0, 0)
            .to_owned()
    });

    image_buffer.save(file_name).unwrap();
}

fn main() -> Result<(), Box<dyn Error>> {
   
    let mut file = File::open("./dataset/mnist_train/file0.csv").expect("you must download the dataset first, https://pjreddie.com/projects/mnist-in-csv/");
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_reader(contents.as_bytes());

    let mut count = 0;
    let mut records: Vec<StringRecord> = vec![];
    for result in reader.records() {
        if count == 100 {
            break;
        }
        let record = result?;
        records.push(record);
        count += 1;
    }

    println!("Records len: {:?}", records.len());
    
    save_record_to_image(&records[0], "./images/five.png");
    save_record_to_image(&records[1], "./images/zero.png");
    save_record_to_image(&records[2], "./images/four.png");


    Ok(())
   
}