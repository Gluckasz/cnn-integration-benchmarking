using Integration.Common;
using Microsoft.Extensions.DependencyInjection;

namespace Integration.ML.NET
{
    public class PredictionServiceFactory(IServiceProvider serviceProvider) : IPredictionServiceFactory
    {
        public IImagePredictionService GetService(string key)
        {
            return serviceProvider.GetKeyedService<IImagePredictionService>(key)
                ?? throw new InvalidOperationException($"Prediction service with key '{key}' not found.");
        }
    }
}
