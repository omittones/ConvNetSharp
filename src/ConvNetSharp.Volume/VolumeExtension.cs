using System;
using System.Collections.Generic;

namespace ConvNetSharp.Volume
{
    public static class VolumeExtension
    {
        public static int IndexOfMax(this Volume<double> output)
        {
            int a = 0;
            for (var i = 1; i < output.Shape.GetDimension(2); i++)
                if (output.Get(0, 0, i, 0) > output.Get(0, 0, a, 0))
                    a = i;
            return a;
        }

        public static void BatchIntoVolume(
           this IEnumerable<Volume<double>> volumes,
           Volume<double> result)
        {
            int count = 0;
            var reshapedResult = result.ReShape(1, 1, -1, result.Shape.GetDimension(3));
            foreach (var volume in volumes)
            {
                if (count >= result.Shape.GetDimension(3))
                    throw new NotSupportedException();

                var reshapedVolume = volume.ReShape(1, 1, -1, 1);
                for (var i = 0; i < reshapedVolume.Shape.TotalLength; i++)
                    reshapedResult.Set(0, 0, i, count, reshapedVolume.Get(0, 0, i, 0));
                count++;
            }
        }
    }
}
