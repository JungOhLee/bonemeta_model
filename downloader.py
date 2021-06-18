import numpy as np
import os
import s3fs
import matplotlib.pyplot as plt
import SimpleITK as sitk
import random

class DataDownloader():
	def __init__(self, remote_dir, local_dir, img_type, label_type, API_key, API_secret, s3bucket):
		self.remote_dir = remote_dir
		self.local_dir = local_dir
		self.img_type = img_type
		self.label_type = label_type
		self.API_key = API_key
		self.API_secret = API_secret

		self.fs = s3fs.S3FileSystem(key = self.API_key, secret = self.API_secret)
		self.bucket = s3bucket
		print(self.bucket, self.remote_dir, self.img_type)
		self.remote_data_location = 's3://{0}'.format(os.path.join(self.bucket, self.remote_dir, self.img_type))
		self.train_list, self.val_list = self.set_train_val_list()

	def set_local_dir(self):
		self.set_dir(self.local_dir)

		self.local_npz_dir = os.path.join(self.local_dir, 'npz')
		self.train_download_npz_dir = os.path.join(self.local_npz_dir, 'train')
		self.val_download_npz_dir = os.path.join(self.local_npz_dir, 'val')
		self.set_dir(self.local_npz_dir)
		self.set_dir(self.train_download_npz_dir)
		self.set_dir(self.val_download_npz_dir)

		self.target_dir = os.path.join(self.local_dir, self.img_type)
		self.target_train_dir = os.path.join(self.target_dir, 'train')
		self.target_val_dir = os.path.join(self.target_dir, 'val')
		self.set_dir(self.target_dir)
		self.set_dir(self.target_train_dir)
		self.set_dir(self.target_val_dir)

	def set_dir(self, path):
		if not os.path.exists(path):
		  os.mkdir(path)

	def set_train_val_list(self):
		filenames = [file['name'] for file in self.fs.listdir(self.remote_data_location)]

		random.seed(3)
		val_list = random.choices(filenames, k = len(filenames)//5 + 1)
		val_list = list(set(val_list))
		train_list = list(set(filenames) - set(val_list))
		return train_list, val_list

	def download_and_unzip(self):
		print("Downloading npz..")
		self.download_npz(self.train_list, self.train_download_npz_dir)
		self.download_npz(self.val_list, self.val_download_npz_dir)
		print("Unzipping npz..")
		self.unzip(self.train_list, self.train_download_npz_dir, self.target_train_dir)
		self.unzip(self.val_list, self.val_download_npz_dir, self.target_val_dir)
		print("Download and unzip completed")

	def download_and_unzip_val(self):
		print("Downloading npz..")
		self.download_npz(self.val_list, self.val_download_npz_dir)
		print("Unzipping npz..")
		self.unzip(self.val_list, self.val_download_npz_dir, self.target_val_dir)
		print("Download and unzip completed")

	def download_npz(self, file_list, download_dir):
		for file in file_list: 
		  self.fs.get(file, os.path.join(download_dir, os.path.split(file)[-1]))

	def unzip(self, npz_file_list, npz_dir, target_dir):
		for file_name in set(npz_file_list):
		  file_path = os.path.join(npz_dir, os.path.split(file_name)[-1])
		  data_and_labels = np.load(file_path)
		  datas = data_and_labels['data']
		  labels = data_and_labels['original_label']
		  for i in range(len(datas)):
		    target_file_name = os.path.splitext(os.path.split(file_name)[-1])[0] + '_{:04d}'.format(i) + '.npy'
		    target_file_path = os.path.join(target_dir, target_file_name)
		    single_data_and_label = np.concatenate([datas[i,:], labels[i,:][np.newaxis, :]], axis=0)
		    np.save(target_file_path, single_data_and_label)
		  # delete unzipped npz file
		  os.remove(file_path)

	def check_sanity(self):
		download_files = self.fs.listdir(self.remote_data_location)
		train_files = os.listdir(self.target_train_dir)
		val_files = os.listdir(self.target_val_dir)

		train_npz = set([filename[:-9] for filename in train_files])
		val_npz = set([filename[:-9] for filename in val_files])
		total_npz_num = len(train_npz) + len(val_npz)

		print("Remote npz_files: {}, Downloaded npz files: {} \n Train npz num: {}, Val npz num: {}".format(
			len(download_files), total_npz_num, len(train_npz), len(val_npz)))
		print("Train npy path: {}, Val npy path:{}".format(self.target_train_dir, self.target_val_dir))

	def set_all(self):
		self.set_local_dir()
		self.download_and_unzip()
		self.check_sanity()

	def set_validation(self):
		self.set_local_dir()
		self.download_and_unzip_val()
		self.check_sanity()