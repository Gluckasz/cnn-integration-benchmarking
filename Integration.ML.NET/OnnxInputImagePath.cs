namespace Integration.ML.NET
{
    public class OnnxInputImagePath(string imagePath)
    {
        public string ImagePath { get; init; } = imagePath;
    }
}
