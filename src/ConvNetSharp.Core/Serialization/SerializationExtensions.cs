using System;
using System.Linq;
using ConvNetSharp.Core.Layers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System.Collections.Generic;

namespace ConvNetSharp.Core.Serialization
{
    public static class SerializationExtensions
    {
        public static string ToHumanString<T>(this T[] items, string itemFormat)
        {
            Func<IEnumerable<T>, string> render = subset => string.Join(", ", subset.Select(i => string.Format(itemFormat, i)));
            if (items.Length > 12)
            {
                var head = items.Take(5);
                var tail = items.Skip(items.Length - 5);
                return $"[{render(head)} ... {render(tail)}]";
            }
            else
            {
                return $"[{render(items)}]";
            }
        }

        public static Net<T> FromJson<T>(string json) where T : struct, IEquatable<T>, IFormattable
        {
            var data = JsonConvert.DeserializeObject<JObject>(json);
            var dico = data.ToDictionary();
            var net = Net<T>.FromData(dico);
            return net;
        }

        public static T[] ToArrayOfT<T>(this object obj)
        {
            var arrayofT = obj as T[];
            if (arrayofT != null)
            {
                return arrayofT;
            }

            var jarray = obj as JArray;
            if (jarray != null)
            {
                return jarray.ToObject<T[]>();
            }

            return ((object[])obj).Select(o => (T)Convert.ChangeType(o, typeof(T), null)).ToArray();
        }

        public static string ToJson<T>(this Net<T> net) where T : struct, IEquatable<T>, IFormattable
        {
            var data = net.GetData();
            var json = JsonConvert.SerializeObject(data);
            return json;
        }

        public static LayerBase<T> FromJsonToLayer<T>(string json) where T : struct, IEquatable<T>, IFormattable
        {
            var data = JsonConvert.DeserializeObject<JObject>(json);
            var dico = data.ToDictionary();
            var layer = LayerBase<T>.FromData(dico);
            return layer;
        }

        public static string ToJson<T>(this LayerBase<T> layer) where T : struct, IEquatable<T>, IFormattable
        {
            var data = layer.GetData();
            var json = JsonConvert.SerializeObject(data);
            return json;
        }
    }
}