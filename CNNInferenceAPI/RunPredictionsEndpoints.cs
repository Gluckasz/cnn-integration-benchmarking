using Integration.Common;

namespace CNNInferenceAPI
{
    public static class RunPredictionsEndpoints
    {
        public static void MapRunPredictionsEndpoints(this WebApplication app)
        {
            app.MapPost(
                "/RunPredictions/MLNET",
                (string[]? imagePaths, IPredictionServiceFactory factory) =>
                {
                    var predictionService = factory.GetService("MLNET");
                    var paths = imagePaths ?? [Constants.DefaultImagesPath];
                    var results = predictionService.Predict(paths);
                    return Results.Ok(results);
                }
            );
        }
    }
}
