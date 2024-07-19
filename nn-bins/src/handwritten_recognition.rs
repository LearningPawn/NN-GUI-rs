use nn::NeuralNetwork;
use image::GenericImageView;
use ndarray::{Array, Dim};
use std::fs::File;
use std::io::prelude::*;
use csv::{ReaderBuilder, StringRecord};
use std::error::Error;
use std::env;

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


fn main() -> Result<(), Box<dyn Error>> {
   
    // arg 1 for train data set Path
    // arg 2 for recognize image Path
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        println!("./handwritten_recognition <train_data_set_path> <recognize_image_path>");
        println!("Example: ./handwritten-digit-recognition ./dataset/mnist_train ./my-own-handwrite-images/handwrite_0.png");
        return Ok(());
    }

    let train_data_set_path = &args[1];
    let recognize_image_path = &args[2];

    let img = image::open(recognize_image_path).unwrap();
    // to grayscale image
    let mut gray_img = img.grayscale();
    // Why invert colorspace? Because the training data gray value is inverted. train set gray value is 0 represent white, 255 represent black.
    // the normal case is 0 represent black, 255 represent white.
    gray_img.invert();
    let gray_img = gray_img.to_luma8();
    let gray_buffer = gray_img.to_vec();


    let (width, height) = gray_img.dimensions();

    if width != 28 || height != 28 {
        println!("The image width and height must be 28");
        return Ok(());
    }

    let image_array: Array<u8, Dim<[usize; 2]>> = Array::from_shape_vec((28, 28), gray_buffer.clone()).unwrap();

    // write image array to txt file
    let mut file = File::create("./images/image_array.txt").unwrap();
    file.write_all(format!("{:?}", image_array).as_bytes()).unwrap();

    let image_vec: Vec<f32> = gray_buffer.iter().map(|x| *x as f32).collect();
    let image_array: Array<f32, Dim<[usize; 2]>> = Array::from_shape_vec((28, 28), image_vec).unwrap();
    // scale the input to range 0.01 to 1.00
    let scaled_input_data = image_array / 255.0 * 0.99 + 0.01;

    // Why output nodes is 10?
    // Because there are 10 digits(0,1,2,3,4,5,6,7,8,9) in the MNIST dataset, so we choose 10 as the output nodes.
    let mut nn = NeuralNetwork::new(28 * 28, 200, 10, 0.1);


    // read full mnist train data file from 0 to 60 for i in range(0, 60]
    println!("Start to train Full mnist the neural network");
    let mut count = 0;

    let file_prefix = format!("{}/file", train_data_set_path);
    while count <= 60 {
        let file_path = format!("{}{}.csv", file_prefix, count);

        let mut file = File::open(&file_path).expect("you must download the dataset first, https://pjreddie.com/projects/mnist-in-csv/");
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;

        let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_reader(contents.as_bytes());

        for result in reader.records() {
            let record = result?;
            
            let image_vec: Vec<u8> = get_img_pixels_data(&record);
            let image_vec: Vec<f32> = image_vec.iter().map(|x| *x as f32).collect();
            let image_array: Array<f32, Dim<[usize; 2]>> = Array::from_shape_vec((28, 28), image_vec).unwrap();
            // scale the input to range 0.01 to 1.00
            let scaled_input_data = image_array / 255.0 * 0.99 + 0.01;

            let mut target_list: Vec<f32> = vec![0.01; 10];
            let real_digit = record.get(0).unwrap();
            let real_digit = real_digit.parse::<u8>().unwrap();
            target_list[real_digit as usize] = 0.99;

            nn.train(&scaled_input_data.iter().cloned().collect(), &target_list);
        }
        count += 1;
    }

    println!("End to train full mnist data the neural network");
    
    let output_list: Vec<f32> = nn.predict(&scaled_input_data.iter().cloned().collect());

    // find the max value's index in the output list
    let mut max_value = 0.0;
    let mut max_index = 0;
    for (index, value) in output_list.iter().enumerate() {
        if *value > max_value {
            max_value = *value;
            max_index = index;
        }
    }

    println!("The recognize image's digit is {}", max_index);

    Ok(())
   
}