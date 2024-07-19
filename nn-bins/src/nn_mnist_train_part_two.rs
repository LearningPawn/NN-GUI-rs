use nn::NeuralNetwork;
use ndarray::{Array, Dim};
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


fn main() -> Result<(), Box<dyn Error>> {
    // Why output nodes is 10?
    // Because there are 10 digits(0,1,2,3,4,5,6,7,8,9) in the MNIST dataset, so we choose 10 as the output nodes.
    let mut nn = NeuralNetwork::new(28 * 28, 200, 10, 0.1);

    // read full mnist train data file from 0 to 60 for i in range(0, 60]
    println!("Start to train Full mnist the neural network");
    let mut count = 0;
    let file_prefix = "./dataset/mnist_train/file";
    while count <= 60 {
        let file_path = format!("{}{}.csv", file_prefix, count);

        println!("Start to train data from file: {:?}", file_path);

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

        println!("End to train data from file: {:?}", file_path);
    }

    println!("End to train full mnist data the neural network");
    
    let mut file = File::open("./dataset/mnist_test.csv").expect("you must download the dataset first, https://pjreddie.com/projects/mnist-in-csv/");
    let mut contents = String::new();
        file.read_to_string(&mut contents)?;
    
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .delimiter(b',')
        .from_reader(contents.as_bytes());

    let mut scored_card: Vec<u64> = vec![];
    for result in reader.records() {
        let record = result?;
        
        let image_vec: Vec<u8> = get_img_pixels_data(&record);
        let image_vec: Vec<f32> = image_vec.iter().map(|x| *x as f32).collect();
        let image_array: Array<f32, Dim<[usize; 2]>> = Array::from_shape_vec((28, 28), image_vec).unwrap();
        // scale the input to range 0.01 to 1.00
        let scaled_input_data = image_array / 255.0 * 0.99 + 0.01;

        let correct_digit = record.get(0).unwrap();
        let correct_digit = correct_digit.parse::<u8>().unwrap();
        
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

        if max_index == correct_digit as usize {
            scored_card.push(1);
        } else {
            scored_card.push(0);
        }
       
    }

    // calculate the performance score, the performance score is the ratio of correct answers to the total number of tests
    let performance_score: f32 = scored_card.iter().sum::<u64>() as f32 / scored_card.len() as f32;
    println!("full mnist train data with 200 hidden layers and 0.1 learning rate result performance score: {}", performance_score);

    Ok(())
   
}