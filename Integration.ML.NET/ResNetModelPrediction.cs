using Microsoft.ML.Data;

namespace Integration.ML.NET
{
    internal class ResNetModelPrediction
    {
        [ColumnName("logits")]
        public float[]? PredictedScores { get; set; }
    }
}
