## Useful SLURM  scripts

#### To get stats for submitted jobs
```
sacct --format="JobID,JobName,Partition,AllocCPUS,State,CPUTime,MaxRSS"
```
#### To get task priority
```
sprio
```
#### To monitor usage amount in seconds
```
sshare -A username
```

#### To send job
```
sbatch filename.sh
```

#### Definition of other command line SLURM functions from TRUBA Wiki (in Turkish)

- ```sinfo``` : İş kuyruklarının güncel kullanım durumunu ekrana basar. Buradan edilinecek bilgi ile kuyruğa gönderilecek işin kaynak miktarı planlanarak en hızlı şekilde başlayabileceği kuyruğa yönlendirilebilir.Kullanılacak ek parametrelerle, listelenecek bilginin türü ve miktarı değiştirilebilir. 

- ```squeue``` : Kullanıcının kuyrukta bekleyen ve çalışan işlerini görüntüler. Kullanılacak ek parametrelerle, listelenecek bilginin türü ve miktarı değiştirilebilir. Kullanıcının tüm işleri listelenebileceği gibi (varsayılan), işin iş numarası parametre olarak verilerek, o işe özel bilgilerin dökümü de alınabilir. 

- ```sbatch``` : Hazırlanan iş betiğini kuyruğa göndermek için kullanılır. Parametreler betik dosyasında verilebilecegi gibi komutun yanına da yazılabilir. İşin akışını, verimini ve kontrolünü sağlayacak pek çok parametresi vardır. 

- ```srun``` : Alternatif olarak işler sbatch yerine srun komutu ile de çalıştırılabilir. Srun kullanılırken çalıştırılacak komut doğrudan srun ve parametrelerinin sonuna yazılır. Basit ve küçük işleri çalıştırmak için hızlı bir yöntemdir. Sbatch'de olduğu gibi pek çok önemli parametresi vardır. 

- ```scancel``` : Kuyrukta sırada bekleyen yada o anda çalışmakta olan işleri iptal etmek için kullanılır. 

- ```salloc``` : Komut satırından interaktif iş çalıştırmak için kullanılır. Salloc komutu ile öncelikle istenilen kaynak miktarı “allocate” edilerek salloc komut satırına geçilir, sonrasında srun komutu işe işler interaktif olarak çalıştırılır. Çalışma sırasında kullanıcı işin çalışmasına müdehale edebilir. 

- ```scontrol``` : Küme, kuyruk (partition) yada herhangi bir iş ile ilgili bilgilerin dökümü alınabilir, izin verildiği ölçüde, müdehale edilebilir. Kuyruğa gönderilmiş olan işler üzerinde, işleri silmeden güncelleme yapılmasına imkan sağlar. 

- ```sacct``` : Beklemede, çalışmakta yada daha önce çalışmış ve sonlanmış olan işler yada tek bir iş hakkında ayrıntılı rapor ve bilgi alınmasına imkan verir. Pek çok parametre içerir. Örneğin belli tarihler arasında başlamış ve bitmiş işlerin listesi, çalışma süresi, kullandığı bellek miktarı, üzerinde çalıştığı sunucuların adresleri vs, gibi iş/işler ile ilgili bilgi alınması mümkündür. 

- ```sstat: ``` Çalışmakta olan işin kullandığı, kullanmakta olduğu sistem kaynakları hakkında bilgi verir. --format= ile verilecek bilgi türleri seçilebilir.

#### Handy scripts:
- Cancel all running/queued tasks for current user: ```squeue -u $USER | awk '{print $1}' | tail -n+2 | xargs scancel```