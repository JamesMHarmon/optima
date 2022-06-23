use std::io::{Write, Seek, SeekFrom, Read};
use anyhow::Result;
use byteorder::{NativeEndian, ReadBytesExt, WriteBytesExt};


pub struct SampleFile {
    sample_len: usize,
}

impl SampleFile {
    pub fn new(sample_len: usize) -> Self {
        Self { sample_len }
    }

    pub fn write<W>(&self, writer: W, samples: &[f32]) -> Result<()>
    where
        W: Write + Seek,
    {
        assert_eq!(samples.len() % self.sample_len, 0);

        let mut writer = writer;

        let num_samples = samples.len() / self.sample_len;

        writer.write_u16::<NativeEndian>(num_samples as u16)?;

        // Move forward past the bytes representing the length of each sample. We don't know their sizes yet
        writer.seek(SeekFrom::Current(num_samples as i64 * 2))?;

        let mut sizes = Vec::with_capacity(num_samples);
        let mut encoded_buf = Vec::with_capacity(self.sample_len);

        for i in 0..num_samples {
            encoded_buf.clear();
            let mut enc = lz4::EncoderBuilder::new()
                .level(4)
                .build(&mut encoded_buf)
                .unwrap();

            let sample_vals = &samples[i * self.sample_len..(i + 1) * self.sample_len];
            let transmuted_vals = bytemuck::cast_slice(sample_vals);

            enc.write_all(transmuted_vals)?;
            enc.finish().1?;

            writer.write_all(&encoded_buf)?;

            sizes.push(encoded_buf.len());
        }

        // Seek back to the beginning + 2 since the first u16 is the number of samples
        writer.seek(SeekFrom::Start(2))?;

        for size in sizes {
            writer.write_u16::<NativeEndian>(size as u16)?;
        }

        Ok(())
    }

    pub fn read<R>(&self, reader: R) -> SampleFileReader<R> {
        SampleFileReader {
            sample_len: self.sample_len,
            reader,
        }
    }
}

pub struct SampleFileReader<R> {
    sample_len: usize,
    reader: R,
}

impl<R> SampleFileReader<R> {
    pub fn num_samples(&mut self) -> Result<usize>
    where
        R: Read + Seek,
    {
        self.reader.seek(SeekFrom::Start(0))?;

        Ok(self.reader.read_u16::<NativeEndian>()? as usize)
    }

    pub fn read_sample(&mut self, sample_idx: usize) -> Result<Vec<f32>>
    where
        R: Read + Seek,
    {
        self.reader.seek(SeekFrom::Start(0))?;

        let num_samples = self.reader.read_u16::<NativeEndian>()? as usize;

        let mut bytes_offset = 2 + (num_samples * 2);
        for _ in 0..sample_idx {
            bytes_offset += self.reader.read_u16::<NativeEndian>()? as usize;
        }

        let sample_num_bytes = self.reader.read_u16::<NativeEndian>()? as usize;

        self.reader.seek(SeekFrom::Start(bytes_offset as u64))?;

        let mut sample_bytes = vec![0; sample_num_bytes];
        self.reader.read_exact(&mut sample_bytes)?;

        let mut dec = lz4::Decoder::new(&*sample_bytes).unwrap();
        let mut sample: Vec<f32> = vec![0.0; self.sample_len];

        let mut sample_mut_ref = bytemuck::cast_slice_mut(sample.as_mut_slice());

        std::io::copy(&mut dec, &mut sample_mut_ref).unwrap();

        Ok(sample)
    }
}

#[cfg(test)]
mod test {
    extern crate test;

    use rand::{prelude::SliceRandom, Rng};
    use std::{
        fs::File,
        io::{BufReader, BufWriter, Write},
        path::PathBuf,
    };

    use super::SampleFile;

    struct FileCleanup(PathBuf);

    impl Drop for FileCleanup {
        fn drop(&mut self) {
            std::fs::remove_file(&self.0).ok();
        }
    }

    fn with_random_file_data<F: FnOnce(BufReader<File>, Vec<f32>, usize)>(func: F) {
        let test_file_path = "./test_cust_file";
        let file_cleanup = FileCleanup(PathBuf::from(&test_file_path));
        let mut rng = rand::thread_rng();
        let num_samples = rng.gen_range(500..1000);
        let sample_len = rng.gen_range(1..((8 * 8 * 17) + 2245 + 1 + 128));

        let data = std::iter::repeat_with(|| (0..sample_len))
            .take(num_samples)
            .flatten()
            .map(|_| rng.gen())
            .collect::<Vec<f32>>();

        let file = File::create(test_file_path).unwrap();
        let mut file = BufWriter::new(file);

        SampleFile::new(sample_len).write(&mut file, &data).unwrap();

        file.flush().unwrap();

        let file = File::open(test_file_path).unwrap();
        let file = BufReader::new(file);

        func(file, data, sample_len);

        drop(file_cleanup);
    }

    #[bench]
    fn bench_read_sample(b: &mut test::Bencher) {
        with_random_file_data(move |_file, _data, sample_len| {
            b.iter(|| {
                let file = File::open("./test_cust_file").unwrap();
                let mut file = BufReader::new(file);
                let res = SampleFile::new(sample_len)
                    .read(&mut file)
                    .read_sample(1)
                    .unwrap();

                std::hint::black_box(res);
            });
        });
    }

    #[test]
    fn file_load_has_correct_data_test() {
        with_random_file_data(|mut file, data, sample_len| {
            let mut rng = rand::thread_rng();

            let mut sample_reader = SampleFile::new(sample_len).read(&mut file);

            let num_samples = sample_reader.num_samples().unwrap();

            let mut sample_nums = (0..num_samples).collect::<Vec<_>>();

            sample_nums.shuffle(&mut rng);

            for idx in sample_nums {
                assert_eq!(
                    &*sample_reader.read_sample(idx).unwrap(),
                    &data[(idx * sample_len)..((idx + 1) * sample_len)]
                );
            }
        });
    }
}
