using Integration.Common;

namespace CNNInferenceAPI
{
    public static class RunPredictionsEndpoints
    {
        public static void MapRunPredictionsEndpoints(this WebApplication app)
        {
            app.MapPost(
                "/RunPredictions/MLNET",
                (string[]? imagePaths, IPredictionServiceFactory factory, Constants constants) =>
                {
                    var predictionService = factory.GetService("MLNET");
                    var paths = imagePaths ?? [constants.DefaultImagesPath];
                    var results = predictionService.Predict(paths);
                    return Results.Ok(results);
                }
            );
        }
    }
}
