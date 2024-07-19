use ndarray::{Array, Dim};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Normal;

/// neural network  struct definition
#[derive(Debug)]
pub struct NeuralNetwork {
    input_nodes: i32,
    hidden_nodes: i32,
    output_nodes: i32,
    learning_rate: f32,
    weight_ih: Array<f32, Dim<[usize; 2]>>, // weights matrix from input to hidden layer
    weight_ho: Array<f32, Dim<[usize; 2]>>, // weights matrix from hidden to output layer
    activation_function: fn(f32) -> f32,

}

impl NeuralNetwork {
    /// Create a new neural network from inputnodes, hiddennodes, outputnodes, learningrate
    pub fn new(inputnodes: i32, hiddennodes: i32, outputnodes: i32, learningrate: f32) -> NeuralNetwork {
        // hiddennodes*inputnodes matrix array
        // mean 0.0 and standard deviation of 1 / sqrt(number of nodes of next layer) = inputnodes^(-0.5)
        let wih = Array::random((hiddennodes as usize, inputnodes as usize),
            Normal::new(0.0, (hiddennodes as f32).powf(-0.5)).unwrap());
        
        // outputnodes*hiddennodes matrix array
        // mean 0.0 and standard deviation of 1 / sqrt(number of nodes of next layer) = hiddennodes^(-0.5)
        let who = Array::random((outputnodes as usize, hiddennodes as usize),
            Normal::new(0.0, (outputnodes as f32).powf(-0.5)).unwrap());

        // 1 / (1 + e^(-x))
        fn sigmoid(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }
        
        NeuralNetwork {
            // set number of nodes in each input, hidden, output layer
            input_nodes: inputnodes,
            hidden_nodes: hiddennodes,
            output_nodes: outputnodes,
            weight_ih: wih,
            weight_ho: who,
            // learning rate
            learning_rate: learningrate,
            // activation function is the sigmoid function
            activation_function: |x| sigmoid(x),
        }
    }

    /// train the neural network
    pub fn train(&mut self, input_list: &Vec<f32>, target_list: &Vec<f32>) {
        let (hidden_output_vec, final_output_vec) = self.forward(input_list);
        let target_vec =  Array::from_shape_vec((target_list.len(), 1), target_list.clone()).unwrap();

        // calculate the error (target - actual_output)
        let output_errors_vec = target_vec - &final_output_vec;
        // calculate the hidden layer errors (errors_hidden = (weights_hidden_to_output^T) * output_errors_vec)
        let hidden_errors_vec = self.weight_ho.t().dot(&output_errors_vec);

        // Update weights for the weight links between the hidden and output layers used With output_errors_vec
        // Delta W = learning_rate * E * O * (1 - O) * hidden_output_vec^T where O is sigmoid(previous_output_vec) and E is next layer errors
        self.weight_ho = &self.weight_ho + self.learning_rate * &output_errors_vec * &final_output_vec * (1.0 - &final_output_vec) * &hidden_output_vec.t();

        // Update weights for the weight links between the input and hidden layers used With hidden_errors_vec
        // Delta W = learning_rate * E * O * (1 - O) * input_vec^T where O is sigmoid(previous_output_vec) and E is next layer errors
        let input_vec = Array::from_shape_vec((input_list.len(), 1), input_list.clone()).unwrap();
        self.weight_ih = &self.weight_ih + self.learning_rate * &hidden_errors_vec * &hidden_output_vec * (1.0 - &hidden_output_vec) * &input_vec.t();

    }

    /// forward pass through the neural network
    pub fn forward(&self, input_list: &Vec<f32>) -> (Array<f32, Dim<[usize; 2]>>, Array<f32, Dim<[usize; 2]>>){
        if input_list.len() != self.input_nodes as usize {
            panic!("input list length does not match input nodes");
        }
        // convert input list to 2d array
        // calculate signals into hidden layer
        let input_vec = Array::from_shape_vec((input_list.len(), 1), input_list.clone()).unwrap();
        let hidden_input_vec = self.weight_ih.dot(&input_vec);
        // calculate the signals emerging from hidden layer
        let hidden_output_vec = hidden_input_vec.mapv(|x| (self.activation_function)(x));

        if hidden_output_vec.shape()[0] != self.hidden_nodes as usize {
            panic!("hidden output vec length does not match hidden nodes");
        }

        // calculate signals into final output layer
        let final_input_vec = self.weight_ho.dot(&hidden_output_vec);
        // calculate the signals emerging from final output layer
        let final_output_vec = final_input_vec.mapv(|x| (self.activation_function)(x));

        if final_output_vec.shape()[0] != self.output_nodes as usize {
            panic!("final output vec length does not match output nodes");
        }

        return (hidden_output_vec, final_output_vec);
    }

    /// query the neural network
    pub fn predict(&self, input_list: &Vec<f32>) -> Vec<f32> {
        let (_, final_output_list)= self.forward(input_list);
        let mut output_list = Vec::new();
        for i in 0..final_output_list.len() {
            output_list.push(final_output_list[[i, 0]]);
        }
        return output_list;
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let mut nn = NeuralNetwork::new(3, 3, 3, 0.3);
        let input_list = vec![1.0, 0.5, -1.5];
        let target_list = vec![0.5, 1.0, 0.5];
        nn.train(&input_list, &target_list);
        let output_list = nn.predict(&input_list);
        println!("{:?}", output_list);
    }
}