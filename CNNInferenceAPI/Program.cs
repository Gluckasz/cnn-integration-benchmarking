using Integration.Common;
using Integration.ML.NET;
using Scalar.AspNetCore;

namespace CNNInferenceAPI
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            builder.Services.AddAuthorization();

            builder.Services.AddOpenApi();

            builder.Services.AddTransient<
                ModelInputBuilder<OnnxInputImagePath>,
                OnnxInputBuilder
            >();
            builder.Services.AddKeyedTransient<
                IImagePredictionService,
                ResNetImagePredictionService
            >("MLNET");
            builder.Services.AddTransient<IPredictionServiceFactory, PredictionServiceFactory>();

            var app = builder.Build();

            if (app.Environment.IsDevelopment())
            {
                app.MapOpenApi();
                app.MapScalarApiReference();
            }

            app.UseAuthorization();

            app.MapRunPredictionsEndpoints();

            app.Run();
        }
    }
}
