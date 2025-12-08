namespace Integration.Common
{
    public record Constants(
        string OnnxModelPath = "..\\Assets\\resnet-50.onnx",
        string DefaultImagesPath = "..\\Assets\\images"
    );
}
