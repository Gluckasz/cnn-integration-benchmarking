using Integration.Common;

namespace Integration.ML.NET
{
    public class ResNetImagePredictionService(ModelInputBuilder<OnnxInputImagePath> builder)
        : IImagePredictionService
    {
        public string[] Predict(string[] images)
        {
            foreach (var image in images)
            {
                builder.WithPath(image);
            }
            var modelInputImages = builder.Build();

            var mlContext = ResNetModelFactory.GetMLContext();
            var model = ResNetModelFactory.GetModel();

            var loadedImages = mlContext.Data.LoadFromEnumerable(modelInputImages);
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
            }

            return output;
        }
    }
}
