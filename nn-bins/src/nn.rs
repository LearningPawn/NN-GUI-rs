use nn::NeuralNetwork;

fn main() {
    let mut nn = NeuralNetwork::new(3, 3, 3, 0.3);
    println!("NN is: {:?}", nn);
    let input_list = vec![1.0, 0.5, -1.5];
    let target_list = vec![0.5, 1.0, 0.5];
    let o = nn.predict(&input_list);
    println!("Output Vector Before train is: {:?}", o);
    nn.train(&input_list, &target_list);
    let o = nn.predict(&input_list);
    println!("Output Vector After train is: {:?}", o);
   
}