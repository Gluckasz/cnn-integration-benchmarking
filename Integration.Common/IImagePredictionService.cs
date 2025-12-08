namespace Integration.Common
{
    public interface IImagePredictionService
    {
        string[] Predict(string[] images);
    }
}
