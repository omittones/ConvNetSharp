using System;
using System.Text;

namespace ConvNetSharp.Volume
{
    public static class ArrayExtensions
    {
        public static string ToString<T>(this T[] arr, string format)
            where T : IFormattable
        {
            var sb = new StringBuilder();
            foreach (var item in arr)
            {
                if (sb.Length != 0)
                    sb.Append(", ");
                sb.Append(item.ToString(format, null));
            }
            return $"[{sb.ToString()}]";
        }

        public static T[] Populate<T>(this T[] arr, T value)
        {
            for (var i = 0; i < arr.Length; i++)
            {
                arr[i] = value;
            }

            return arr;
        }
    }
}