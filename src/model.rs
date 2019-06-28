pub trait Model {
    fn get_name(&self) -> &str;
}

pub trait ModelFactory
{
    type M: Model;

    fn create(&self, name: &str) -> Self::M;
    fn get_latest(&self, name: &str) -> Self::M;
    fn train(&self, from_name: &str, target_name: &str, options: &TrainOptions) -> Self::M;
}

pub struct TrainOptions {

}
