using Integration.Common;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;

namespace Integration.ML.NET
{
    public class ResNetImagePredictionService(
        string modelPath,
        ModelInputBuilder<OnnxInputImagePath> builder
    ) : IImagePredictionService
    {
        private static readonly int _imageWidth = 224;
        private static readonly int _imageHeight = 224;

        private readonly int[] _outputShape = [1, 1000];
        private readonly int[] _inputShape = [1, 3, _imageWidth, _imageHeight];

        public string[] Predict(string[] images)
        {
            foreach (var image in images)
            {
                builder.WithPath(image);
            }
            var modelInputImages = builder.Build();

            var mlContext = new MLContext();

            var loadedImages = mlContext.Data.LoadFromEnumerable(modelInputImages);

            var pipeline = mlContext
                .Transforms.LoadImages(
                    outputColumnName: "Image",
                    imageFolder: "",
                    inputColumnName: nameof(OnnxInputImagePath.ImagePath)
                )
                .Append(
                    mlContext.Transforms.ResizeImages(
                        outputColumnName: "resized_image",
                        imageWidth: _imageWidth,
                        imageHeight: _imageHeight,
                        inputColumnName: "Image"
                    )
                )
                .Append(
                    mlContext.Transforms.ExtractPixels(
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
                    mlContext.Transforms.ApplyOnnxModel(
                        outputColumnNames: ["logits"],
                        inputColumnNames: ["pixel_values"],
                        modelFile: modelPath,
                        shapeDictionary: new Dictionary<string, int[]>
                        {
                            { "pixel_values", _inputShape },
                            { "logits", _outputShape },
                        },
                        gpuDeviceId: null,
                        fallbackToCpu: true
                    )
                );

            var model = pipeline.Fit(loadedImages);
            var transformedData = model.Transform(loadedImages);

            var predictions = mlContext.Data.CreateEnumerable<ResNetModelPrediction>(
                transformedData,
                reuseRowObject: false
            );

            List<ResNetModelPrediction> predictionsList = [.. predictions];

            var output = new string[modelInputImages.Count];

            for (var i = 0; i < predictions.Count(); i++)
            {
                var topPrediction = predictionsList[i]
                    .PredictedScores!.Select((score, index) => new { Score = score, Index = index })
                    .OrderByDescending(x => x.Score)
                    .First();

                output[i] = topPrediction.Index.ToString();
                Console.WriteLine(topPrediction.Index.ToString());
            }

            return output;
        }
    }
}
