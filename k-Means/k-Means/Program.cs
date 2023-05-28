using System;
using System.Collections.Generic;
using System.IO;
using Accord.MachineLearning;
using Accord.Math;
using Accord.Controls;

namespace k_Means
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Verilerin okunması
            using (var okuma = new StreamReader("Mall_Customers.csv"))
            {
                var veri = new List<double[]>();

                // Başlıkları veri olmadığı için atlanır.
                okuma.ReadLine();

                // Veriler okunarak diziye yazdırılır.
                while (!okuma.EndOfStream)
                {
                    var satir = okuma.ReadLine();
                    var deger = satir.Split(',');

                    var yillikGelir = Convert.ToDouble(deger[2]);
                    var harcamaPuani = Convert.ToDouble(deger[3]);

                    veri.Add(new double[] { yillikGelir, harcamaPuani });
                }
                // Okunan veriler iki boyutlu diziye aktarılır.
                double[][] dizi = veri.ToArray();

                // Elbow Metoduyla Küme Sayısını Belirleme
                int[] kDegerleri = new int[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
                double[] karelerToplami = new double[kDegerleri.Length];

                for (int i = 0; i < kDegerleri.Length; i++)
                {
                    int k = kDegerleri[i];

                    var kmeans = new KMeans(k);
                    kmeans.Tolerance = 0.05;

                    var kumeler = kmeans.Learn(dizi);

                    double karelerToplam = 0;
                    for (int j = 0; j < dizi.Length; j++)
                    {
                        int kume = kumeler.Decide(dizi[j]);
                        double gecici = Math.Pow(Distance.Euclidean(dizi[j], kmeans.Clusters.Centroids[kume]), 2);
                        karelerToplam += gecici;
                    }

                    karelerToplami[i] = karelerToplam;
                }

                Console.WriteLine("K Değerleri\tUzunlukların karelerinin toplamı");
                for (int i = 0; i < kDegerleri.Length; i++)
                {
                    Console.WriteLine(kDegerleri[i] + "\t\t" + karelerToplami[i]);
                }

                // K-Means Kümeleme
                int optimalK = 5;
                var kmeansOptimal = new KMeans(optimalK);
                kmeansOptimal.Tolerance = 0.05;

                var optimalKumeler = kmeansOptimal.Learn(dizi);

                // Grafik Oluşturma
                ScatterplotBox.Show("Müşteri Segmantasyonu", dizi, optimalKumeler.Decide(dizi));

                // Yazdırma
                Console.WriteLine("\nİdeal K: " + optimalK);
                Console.WriteLine("Merkezler:");
                for (int i = 0; i < optimalK; i++)
                {
                    Console.WriteLine("(" + kmeansOptimal.Clusters.Centroids[i][0] + "," + kmeansOptimal.Clusters.Centroids[i][1] + ")");
                }

                Console.WriteLine("\nKüme Atamaları:");
                for (int i = 0; i < dizi.Length; i++)
                {
                    int kume = optimalKumeler.Decide(dizi[i]);
                    Console.WriteLine("(" + dizi[i][0] + ", " + dizi[i][1] + ") - Kume: " + kume);
                }
            }
        }
    }
}
