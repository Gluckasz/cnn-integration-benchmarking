namespace Integration.Common
{
    public interface IPredictionServiceFactory
    {
        IImagePredictionService GetService(string key);
    }
}
