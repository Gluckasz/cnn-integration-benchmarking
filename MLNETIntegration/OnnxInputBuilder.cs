namespace Integration.ML.NET
{
    public class OnnxInputBuilder : ModelInputBuilder<OnnxInputImagePath>
    {
        public override List<OnnxInputImagePath> Build()
        {
            return [.. _modelInputImagePaths.Select(path => new OnnxInputImagePath(path))];
        }
    }
}
