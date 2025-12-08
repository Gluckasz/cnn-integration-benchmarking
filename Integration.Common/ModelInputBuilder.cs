namespace Integration.Common
{
    public abstract class ModelInputBuilder<T>
    {
        protected readonly List<string> _modelInputImagePaths = [];

        public void WithPath(string path)
        {
            ValidatePath(path);

            FileAttributes pathAttributes = File.GetAttributes(path);

            if (pathAttributes.HasFlag(FileAttributes.Directory))
            {
                ProcessDirectory(path);
            }
            else
            {
                ValidateImageExtension(path);
                _modelInputImagePaths.Add(path);
            }
        }

        public abstract List<T> Build();

        private static void ValidatePath(string path)
        {
            if (!File.Exists(path) && !Directory.Exists(path))
            {
                throw new FileNotFoundException(
                    $"The path {path} must lead to existing file or directory."
                );
            }
        }

        private void ProcessDirectory(string directoryPath)
        {
            var imagePaths = Directory.GetFiles(directoryPath);
            foreach (var imagePath in imagePaths)
            {
                ValidateImageExtension(imagePath);
                _modelInputImagePaths.Add(imagePath);
            }
        }

        private static void ValidateImageExtension(string imagePath)
        {
            var postedFileExtension = Path.GetExtension(imagePath);
            if (
                !string.Equals(postedFileExtension, ".jpg", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(postedFileExtension, ".png", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(postedFileExtension, ".jpeg", StringComparison.OrdinalIgnoreCase)
            )
            {
                throw new FileLoadException(
                    $"The file {imagePath} must have .jpg, .png, or .jpeg extension."
                );
            }
        }
    }
}
