# Laporan Proyek Machine Learning - Karlina Surya Witanto

## Domain Proyek
Domain yang dieksplorasi dalam proyek kali ini adalah finance atau finansial. Secara spesifik fokus finansial pada
proyek ini adalah berkaitan dengan pergerakan harga saham, dalam kasus ini kita menggunakan saham dari perusahaan tesla (TSLA) yang
dapat ditemukan di [nasdaq - TSLA](https://www.nasdaq.com/market-activity/stocks/tsla/historical).
Banyaknya permintaan dari para pedagang atau trader saham yang membutuhkan informasi mengenai ramalan dari harga saham di masa depan.
Jadi disinilah sebuah metode yaitu prediksi harga saham dilakukan. Metode ini merupakan sebuah cara untuk menentukan/meramal nilai
saham dari sebuah perusahaan di masa depan, prediksi yang mendekati 100% dapat memberikan profit yang sangat besar.

Domain spesifik ini dapat dicapai dengan cara menggunakan analisis teknikal menggunakan informasi dari perusahaan itu sendiri
serta melibatkan intuisi dalam meramalnya, tetapi metode yang lebih tepat adalah menggunakan perantara machine learning atau
pembelajaran mesin yang dapat melakukan prediksi harga saham dengan tingkat akurasi yang sangat tinggi.

Hal tersebut terbukti dari banyaknya penelitian yang membuktikan kemampuan machine learning dalam memprediksi harga saham, diantaranya:
- [Reddy et al.](https://www.academia.edu/download/57764282/IRJET-V5I10193.pdf)
- [Sharma et al.](https://ieeexplore.ieee.org/abstract/document/8212715/)
- Lebih spesifik ada yang menggunakan deep learning yaitu [Selvin et al.](https://www.sciencedirect.com/science/article/pii/S1877050920304865)


## Business Understanding
Masalah yang paling sering di hadapi para trader adalah dalam jual beli saham merupakan ketidakpastian dari pergerakan saham di masa depan.
Perusahaan yang memberitahukan nilai dari harga sahamnya juga tergantung dari image dan pemasarannya itu seperti apa itu sifatnya sangat
dinamis. Mekanisme dari penentuan harga saham berdasarkan harga yang disetujui dari pembeli dan penjual. Jika pembeli lebih banyak daripada
penjual maka harga dari sebuah saham akan naik, sebaliknya harganya bisa turun.

Dengan menggunakan pendekatan machine learning yang dilatih menggunakan data historical dari harga saham suatu perusahaan dalam hal kasus
ini adalah tesla (TSLA) akan membantu dalam proses pemerolehan informasi berdasarkan prediksinya apakah dimasa yang akan datang sebagai
contoh 7 hari kedepan itu ditiap harinya pergerakan saham akan naik, turun atau tetap.

### Problem Statements
- Apakah machine learning dapat diimplementasikan untuk memprediksi harga saham?
- Bagaimana performa dari perbanding 3 model machine learning yaitu LSTM, Linear Regression dan Support Vectore Regression
dalam memprediksi harga saham?

### Goals
- Dapat mengimplementasikan model machine learning yang dilatih menggunakan data lampau serta memprediksi harga saham.
- Mengevaluasi performa dari machine learning dan membandingkannya satu sama lain menggunakan metrik evaluasi regresi.

### Solution statements
Dalam kasus ini kita memiliki 3 model yang berbeda, dari yang sederhana sampai kompleks, diantaranya:
- **Linear Regression**. Model ini merupakan algoritma yang melakukan pemodelan dari hubungan dua variabel dengan proses fitting 
equasi linear terhadap data yang diobservasi. Model ini digunakan untuk memprediksi nilai suatu variable berdasarkan nilai dari variable
lainnya. Nilai yang diprediksi merupakan variabel yang dependen, sedangkan nilai variable yang digunakan untuk memprediksi merupakan 
variabel independen. 
- **Support Vector Regression**. Mekanisme dari SVR ini yaitu dengan mencari garis fit terbaik atau sebuah hyperplane yang dianggap
paling optimal dalam memprediksi nilai-nilai dari data kontinu dengan memetakan data yang terurut dengan pemisahan data berdasarkan
margin (hyperplane) terbaik.
- **Long Short-Term Memory**. Model ini berbasis jaringan syaraf tiruan dan varian dari recurrent neural network (RNN).
yang berkemampuan untuk mempelajari urutan kebergantungan dari sebuah data sekuensial
- Mengevaluasi performa dari machine learning dan membandingkannya satu sama lain menggunakan metrik evaluasi regresi.

## Data Understanding
Di bidang finansial, data pasar adalah harga dan data terkait lainnya untuk instrumen keuangan yang dilaporkan oleh tempat perdagangan seperti bursa saham. 
Data pasar memungkinkan pedagang dan investor mengetahui harga terbaru dan melihat tren historis untuk instrumen seperti ekuitas, produk pendapatan tetap, 
derivatif, dan mata uang. Seperti dalam kasus ini kita menggunkan [Dataset Tesla](https://www.nasdaq.com/market-activity/stocks/tsla/historical).
yang memiliki rentang waktu 5 tahun yaitu dari 2016 hingga 2021. Data tersebut berisi informasi harga saham dari perusahaan tesla dari rentang tahun tersebut
dan kita menemukan kenaikan yang signifikan dari saham tersebut sejak tahun 2016. Jumlah data total yaitu 1259 baris dengan atribut atau indikator saham 
yang memiliki penanggalan berdasarkan harinya. 

Selanjutnya, data lampau dari harga saham ini terdiri dari beberapa variabel sebagai berikut:
- Date : merupakan sebuah variabel yang berisi data tentang tanggal dari harga spesifik suatu saham.
- Close : merupakan harga terakhir di mana sekuritas diperdagangkan selama hari perdagangan reguler.
- Volume : Volume dihitung sebagai jumlah total saham yang benar-benar diperdagangkan (dibeli dan dijual) selama hari perdagangan atau 
periode waktu yang ditentukan.
- Open : merupakan harga di mana saham mulai diperdagangkan saat bel pembukaan berbunyi. 
Bisa sama seperti dimana stock tutup malam sebelumnya.
- High : merupakan harga tertinggi di mana suatu saham diperdagangkan selama suatu periode.
- Low : merupakan harga terendah di mana suatu saham diperdagangkan selama suatu periode.

Dari kondisi data tersebut kita melakukan perbaikan dalam format penanggalannya. Data tersebut dilakukan proses Exploratory Data Analysis untuk menganalisis 
pergerakan saham dari tahun 2016 hingga tahun 2020 menggunakan library matplotlib. Pengaturan format penanggalan yaitu kolom Data harus sesuai dengan tipe 
data date dengan format tanggal yang bisa digunakan untuk plot data dan sebagai index dari data rangkaian waktu tersebut. Mengurutkan data awal ke akhir dari 
paling lampau sampai paling terkini agar terurut dengan benar index harga dari saham tersebut.
 
## Data Preparation
Proses untuk mempersiapkan data terdiri dari beberapa proses sebagai berikut :
- Normalisasi yaitu untuk mengubah nilai kolom numerik dalam kumpulan data ke skala umum, tanpa mendistorsi perbedaan dalam rentang nilai.
- Splitting data yaitu memisahkan data untuk proses latih dan proses untuk menguji kemampuan model yang akan di validasi mengunakan metrik regresi.

## Modeling
Untuk bagian proses ini adalah memodelkan machine learning dengan data yang telah kita siapkan melalui beberapa proses di Data Preparation.
Ada 3 model kita gunakan pada proses ini yaitu Linear Regression, SVR dan LSTM. Dari hasil prediksi model-model tersebut kita memperoleh Linear regression
dengan nilai mean absolute error (nilai error) paling rendah diikuti dengan SVR pada posisi kedua dan LSTM pada posisi ketiga menggunakan data yang jumlahnya
lebih dari 1000 buah dengan rentang waktu 2016 sampai 2021.

Linear Regression menggunakan default parameter boolean terbaik untuk datanya dengan memberikan error MAE sebesar : 0.0034310297676294938 dengan
akurasi menggunakan R2 Score (Coefficient Determination) sebesar : 0.9994044067327629, dari 2 metrik ini mengindikasikan bahwa hasil prediksi dari 
Linear Regresion lebih akurat dibandingkan dari LSTM dan SVR.

SVR berada ada posisi ketiga dalam perbandingan ini dengan nilai MAE : 0.07784264473570811 serta akurasi R2 Score (Coefficient Determination) sebesar : 0.9237742811000287, dengan nilai seperti ini sebenarnya dalam analisis hasil prediksi tersebut, performa SVR juga sudah dianggap sangat bagus, tapi dalam pemahaman business yang mana membutuhkan hasil yang sebisa mungkin sangat dekat nilai prediksinya dengan tren asli di masa depan untuk mendapatkan keuntungan
yang maksimum dalam penjualan saham. Model ini juga sebelumnya memiliki performa yang rendah sebelum melalui proses hyperparameter tuning, tetapi setelah
dilakukan tuning tersebut menggunakan parameter berikut :
- C = 0.5 
- kernel = 'linear'
- degree = 10 
- coef0=1 
- tol=1e-3
- cache_size=4096
- gamma=0.7
Nilai parameter inilah yang cukup tepat untuk meningkatkan performa dari model SVR khusus untuk menangani data yang dipakai untuk melatih dan mengujinya 
sehingga bisa memperoleh hasil dari nilai error dan akurasi yang telah disebutkan diatas.


LSTM sendiri memperoleh hasil nilai error MAE : 0.7 pada validasi data ujinya dengan akurasi R2-Squared (Coefficient Determination) sekitar 0.30 yang mengindikasikan bahwa kemampuan LSTM ini sangat rendah walaupuntelah melalui beberapa improvement baseline yang sebelumnya menggunakan 1 layer dengan 128 unit neuron dengan nilai error yang masih tinggi. Kemudian dilakukan proses hyperparameter tuning dengan menambahkan 2 buah layer lstm dengan 64 dan 32 unit neuron masing-masing, kemudian ada penambahan dense layer untuk meningkatkan kompleksitas jaringannya untuk menangani rangkaian data tersebut. Hasil ini masih bisa ditingkatkan lagi karena keterbukaan jaringan syaraf tiruan yang bisa selalu ditingkatkan performanya dengan menambahkan parameter regularisasi serta jumlah unit yang berbeda sebagai faktor yang memungkinkan hasilnya bisa menjadi lebih baik dibanding SVR atau bahkan linear regression.



## Evaluation
- R2 Score

R2-Squared adalah ukuran statistik tentang seberapa dekat data dengan garis regresi yang sesuai dan bisa disebut sebagai performa seberapa akurat
model machine learning dalam memprediksi data uji.

![R-Squared](https://miro.medium.com/max/1200/1*_HbrAW-tMRBli6ASD5Bttw.png)

Perhitungan R-squared yang sebenarnya membutuhkan beberapa langkah. Ini termasuk mengambil titik data (pengamatan) dari variabel dependen dan independen dan menemukan garis yang paling cocok, seringkali dari model regresi. Dari sana Anda akan menghitung nilai prediksi, mengurangi nilai aktual, dan mengkuadratkan hasilnya. Ini menghasilkan daftar kesalahan yang dikuadratkan, yang kemudian dijumlahkan dan sama dengan varians yang tidak dapat dijelaskan.

Untuk menghitung varians total, Anda akan mengurangi nilai aktual rata-rata dari masing-masing nilai aktual, kuadratkan hasilnya, dan jumlahkan. Dari sana, bagi jumlah kesalahan pertama (varian yang dijelaskan) dengan jumlah kedua (varian total), kurangi hasilnya dari satu, dan itulah hasil R-kuadrat.

R-Squared hanya berfungsi sebagaimana dimaksud dalam model regresi linier sederhana dengan satu variabel penjelas. Dengan regresi berganda yang terdiri dari beberapa variabel bebas, maka R-Squared harus disesuaikan.

Sumber : [tautan](https://www.google.com/url?sa=i&url=https%3A%2F%2Fstats.stackexchange.com%2Fquestions%2F183265%2Fwhat-does-negative-r-squared-mean&psig=AOvVaw2ERqw4b6Et0SYYSDp3ER_A&ust=1633853830870000&source=images&cd=vfe&ved=2ahUKEwizq_P68bzzAhXTm0sFHdLCBIMQr4kDegUIARDYAQ)

Dan cara mengimplementasikannya dalam kode adalah :

```
from sklearn.metrics import r2_score

print('R2 score (Coefficient of Determination):',r2_score(truth,prediction))
```

- Mean Absolute Error (MAE)

MAE adalah metrik yang sangat sederhana yang menghitung perbedaan absolut antara nilai aktual dan prediksi. 
MAE model Anda pada dasarnya adalah kesalahan yang dibuat oleh model yang dikenal sebagai kesalahan atau eror. 
Sekarang temukan perbedaan antara nilai aktual dan nilai prediksi yang merupakan kesalahan absolut tetapi kita harus menemukan mean absolut dari kumpulan data lengkap. Jadi, jumlahkan semua kesalahan dan bagi dengan jumlah total pengamatan Dan ini adalah MAE. Dan kami bertujuan untuk mendapatkan MAE minimum karena ini adalah error atau kerugian.

Mengingat setiap kumpulan data pengujian, Mean Absolute Error model yang dievaluasi mengacu pada rata-rata nilai absolut dari setiap kesalahan prediksi pada semua instance 
dari kumpulan data pengujian. Kesalahan prediksi adalah perbedaan antara nilai aktual dan nilai prediksi untuk contoh itu. Secara statistik, Mean Absolute Error (MAE) mengacu pada 
hasil pengukuran perbedaan antara dua variabel kontinu. Jika asumsikan variabel M dan N mewakili fenomena yang sama tetapi telah mencatat pengamatan yang berbeda.
Untuk plot pencar yang diberikan titik x, di mana titik j memiliki koordinat (Mj, Nj). Mean Absolute Error (MAE) disini yang akan menjadi jarak vertikal rata-rata antara setiap titik dan garis N=M. 
Ini juga dikenal sebagai garis One-to-One. 
MAE juga pada titik ini akan menjadi rata-rata jarak horizontal total antara setiap titik dan garis N=M.

![Mean Absolute Error](https://miro.medium.com/max/1400/1*mmUJKWuxJL56AudpnC5qYQ.png)

Kelebihan MAE : 
- MAE yang Anda dapatkan berada di unit yang sama dengan variabel output.
- MAE paling kuat untuk menangani outlier.

Kekurangan MAE yaitu grafik yang tidak terdiferensiasi sehingga kita harus menerapkan berbagai pengoptimal seperti penurunan Gradien yang dapat didiferensiasikan.

Dan cara mengimplementasikanya dalam kode adalah :
```
from sklearn.metrics import mean_absolute_error

print("MAE :",mean_absolute_error(truth,prediction))
```
Khusus untuk tensorflow cara menggunakan MAE seperti ini :
```
#Compile menggunakan metrics MAE
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
```

Metrik-metrik inilah yang sesuai untuk mengevaluasi konteks data kita yang kontinu serta menjawab performa dari problem statement kita untuk melatih
model machine learning dalam melakukan regresi dan angka tersebut menjawab solusi yang kita butuhkan yaitu model apa yang cocok dalam melakukan prediksi
harga saham tesla (TSLA) tersebut, dan kesimpulannya kita menemukan bahwa Linear Regression lebih cocok dalam kasus ini.





