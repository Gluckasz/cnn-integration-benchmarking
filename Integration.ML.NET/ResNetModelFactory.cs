using Integration.Common;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;

namespace Integration.ML.NET
{
    public static class ResNetModelFactory
    {
        private static readonly int _imageWidth = 224;
        private static readonly int _imageHeight = 224;

        private static readonly int[] _outputShape = [1, 1000];
        private static readonly int[] _inputShape = [1, 3, _imageWidth, _imageHeight];

        private static readonly MLContext _mlContext = new();

        private static readonly Lazy<ITransformer> _model = new(() =>
        {
            var dummyData = _mlContext.Data.LoadFromEnumerable([new OnnxInputImagePath("")]);

            var pipeline = _mlContext
                .Transforms.LoadImages(
                    outputColumnName: "Image",
                    imageFolder: "",
                    inputColumnName: nameof(OnnxInputImagePath.ImagePath)
                )
                .Append(
                    _mlContext.Transforms.ResizeImages(
                        outputColumnName: "resized_image",
                        imageWidth: _imageWidth,
                        imageHeight: _imageHeight,
                        inputColumnName: "Image"
                    )
                )
                .Append(
                    _mlContext.Transforms.ExtractPixels(
                        outputColumnName: "pixel_values",
                        inputColumnName: "resized_image",
                        colorsToExtract: ImagePixelExtractingEstimator.ColorBits.Rgb,
                        orderOfExtraction: ImagePixelExtractingEstimator.ColorsOrder.ABGR,
                        interleavePixelColors: false,
                        offsetImage: 0f,
                        scaleImage: 1f / 255f
                    )
                )
                .Append(
                    _mlContext.Transforms.ApplyOnnxModel(
                        outputColumnNames: ["logits"],
                        inputColumnNames: ["pixel_values"],
                        modelFile: Constants.OnnxModelPath,
                        shapeDictionary: new Dictionary<string, int[]>
                        {
                            { "pixel_values", _inputShape },
                            { "logits", _outputShape },
                        },
                        gpuDeviceId: null,
                        fallbackToCpu: true
                    )
                );

            return pipeline.Fit(dummyData);
        });

        public static ITransformer GetModel() => _model.Value;

        public static MLContext GetMLContext() => _mlContext;
    }
}
