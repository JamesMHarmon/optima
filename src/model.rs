pub trait Model {
    fn create(&mut self, name: &str);
    fn train(&mut self, from_name: &str, target_name: &str, options: &TrainOptions);
}

pub struct TrainOptions {

}
