import ast
import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer


def data_transform_func():
    df = pd.read_csv('/opt/airflow/data_csv/tangerang.csv')
    df = df[~(df.spesifikasi =='spesifikasi')]
    df['spesifikasi'] = df.spesifikasi.apply(ast.literal_eval)
    df = df.explode('spesifikasi')

    df['keterangan'] = df.spesifikasi.str[0]
    df['qty'] = df.spesifikasi.str[1]

    df_columns = df[['harga','alamat','fasilitas']].reset_index().drop_duplicates('index').set_index('index')
    df_pivot = df.pivot(columns='keterangan', values='qty').rename_axis(None, axis=1)
    df = pd.concat([df_columns, df_pivot], axis = 1)
    df.columns = df.columns.map(lambda x: x.replace(' ', '_'))

    return df

def data_pbi_func(df):
    df = df.drop_duplicates('id_iklan').reset_index(drop=True)

    df.insert(3, 'komplek', df[['fasilitas']].fillna('none').apply(lambda z: 'ya' if any([x in z.fasilitas.lower() for x in ['lapangan','gym','jogging','playground','one gate system']]) else 'tidak', axis = 1))
    df.insert(1, 'Kecamatan', df.alamat.str.split(',').str[0])
    df.insert(2, 'Kota', df.alamat.str.split(',').str[1])
    df.drop(columns=['alamat'], inplace=True)

    df = df[df.Kota == ' Tangerang']
    df.drop(columns = ['Kota'], inplace=True)

    def perabot(kondisi):
        kondisi = str(kondisi)
        if kondisi.lower() in ['unfurnished','butuh renovasi']:
            return 'Unfurnished'
        elif kondisi.lower() in ['furnished','bagus','bagus sekali','baru','sudah renovasi','semi furnished']:
            return 'Furnished'
        else:
            return np.nan
    df['kondisi_perabotan'] = df['kondisi_perabotan'].map(lambda x: perabot(x))

    df = df[df['tipe_properti'] == 'Rumah']

    def price_extract(price):
        if "Triliun" in price:
            numbers = re.findall(r'\d+\.\d+|\d+', price)
            numbers = float(".".join(numbers))
            return int(numbers * 1000000)
        elif "Miliar" in price:
            numbers = re.findall(r'\d+\.\d+|\d+', price)
            numbers = float(".".join(numbers))
            return int(numbers * 1000)
        elif "Juta" in price:
            numbers = re.findall(r'\d+\.\d+|\d+', price)
            numbers = float(".".join(numbers))
            return int(numbers * 1)      

    df.insert(1, 'price',  df[['harga']].apply(lambda x: price_extract(x.harga), axis = 1))
    df.drop(columns = ['harga'], inplace = True)
    df.insert(3, 'ID',  df['id_iklan'])
    df = df[df.sertifikat == 'SHM - Sertifikat Hak Milik']
    df['carport/Garage'] = (~((df['carport'].isna()) & (df['garasi'].isna()))).astype(int)
    df['komplek'] = df['komplek'].map(lambda x: 0 if x.lower() == 'tidak' else 1)
    df = df[~df['luas_tanah'].isna()].reset_index(drop=True)

    for kol in ['kamar_mandi','kamar_tidur']:
        df[kol] = df[kol].fillna(1)

    df['luas_bangunan'] = df['luas_bangunan'].fillna(df['luas_tanah'])
    df['kamar_mandi_pembantu'] = df['kamar_mandi_pembantu'].fillna(0)
    df['kamar_pembantu'] = df['kamar_pembantu'].fillna(0)

    def conditions(x):
        if pd.isnull(x['jumlah_lantai']):
            if x['luas_bangunan'] > x['luas_tanah']:
                return '2'
            else:
                return '1'
        else:
            return x['jumlah_lantai']

    df['jumlah_lantai'] = df[['jumlah_lantai','luas_bangunan','luas_tanah']].apply(lambda x: conditions(x) , axis = 1 )

    condition = ((df['daya_listrik'] == 'Lainnya Watt') | (df['daya_listrik'] == 'lainnya Watt'))
    df.loc[condition, 'daya_listrik'] = np.nan

    df['daya_listrik'] = df['daya_listrik'].str.replace('Watt','').astype('Float64')
    df['luas_bangunan'] = df['luas_bangunan'].str.replace('m²','').astype(int)
    df['luas_tanah'] = df['luas_tanah'].str.replace('m²','').astype(int)

    df['avg_bangunan'] = df.groupby('daya_listrik')['luas_bangunan'].transform('median').fillna(df['luas_bangunan'])
    listrik_impute = pd.DataFrame(KNNImputer(n_neighbors=1).fit_transform(df[['daya_listrik','avg_bangunan']]))
    df['daya_listrik'] = listrik_impute[0]
    df.drop(columns=['avg_bangunan'], inplace=True)

    value_mapping = {
    'Unfurnished': 1,
    'Furnished' : 2,
    }

    reverse_mapping = {
    1.0 : 'Unfurnished',
    2.0 : 'Furnished',
    }

    for kec in df.Kecamatan.unique() :
        df1 = df[df.Kecamatan == kec][['ID','kondisi_perabotan','luas_tanah','luas_bangunan','price']].copy().reset_index(drop=True)
        df1['harga/m2'] = df1.price / (df1['luas_tanah'] + df1['luas_bangunan'])
        df1['hargagroup'] = df1.groupby(['kondisi_perabotan'])['harga/m2'].transform('median').fillna(df1['harga/m2'])
        df1['kondisi_perabotan'] = df1['kondisi_perabotan'].map(value_mapping).astype('Int64')
        
        perabot_impute = pd.DataFrame(KNNImputer(n_neighbors=1).fit_transform(df1[['kondisi_perabotan','hargagroup']]))
        df1['kond_perabot'] = perabot_impute[0]

        df.set_index('ID', inplace=True, drop=False)
        df1.set_index('ID', inplace=True, drop=False)

        df['kondisi_perabotan'] = df['kondisi_perabotan'].combine_first(df1['kond_perabot'])

        df.reset_index(drop=True, inplace=True)


    df['kondisi_perabotan'] = df['kondisi_perabotan'].replace(reverse_mapping)

    df.drop(columns=['fasilitas','tahun_di_renovasi','tahun_dibangun','id_iklan','sumber_air','ruang_tamu','ID','hook','carport','garasi','material_bangunan','material_lantai','dapur','hadap','konsep_dan_gaya_rumah','lebar_jalan','nomor_lantai','pemandangan','Periode Sewa','ruang_makan','sertifikat','terjangkau_internet','tipe_properti'], errors='ignore',inplace=True)

    def properti(kondisi):
        kondisi = str(kondisi)
        if kondisi.lower() in ['unfurnished','butuh renovasi']:
            return 'Butuh renovasi'
        elif kondisi.lower() in ['furnished','bagus','bagus sekali','baru','sudah renovasi','semi furnished']:
            return 'Bagus'
        else:
            return np.nan

    df['kondisi_properti'] = df['kondisi_properti'].map(lambda x: properti(x))
    df['kondisi_properti'] = df['kondisi_properti'].fillna('Bagus')

    df = df.convert_dtypes(convert_string = False)
    for column in ['kamar_mandi','kamar_mandi_pembantu','kamar_pembantu','kamar_tidur','price','daya_listrik','luas_bangunan','luas_tanah']:
        df[column] = df[column].astype(int)

    df.rename(columns={'daya_listrik': 'Listrik', 
                   'jumlah_lantai': 'Lantai',
                   'kamar_mandi': 'KM',
                   'kamar_mandi_pembantu': 'KMP',
                   'kamar_pembantu': 'KP',
                   'kamar_tidur': 'KT',
                   'kondisi_perabotan': 'Kondisi',
                   'luas_bangunan': 'LB',
                   'luas_tanah': 'LT'}, inplace=True)
    
    df = df.drop_duplicates()
    df = df[df['Lantai'].astype(int) <= 4]

    df = df[(df.price < 10000)]
    df = df[(df['Listrik'] < 20000)]
    df = df[(df['KM'] < 10)]
    df = df[(df['KMP'] <= 2)]
    df = df[(df['KP'] <= 2)]
    df = df[(df['KT'] <= 20)]
    df = df[(df['LB'] <= 1000)]
    df = df[(df['LT'] <= 1000)]

    return df


def data_evidently_func(df):
    df = df.drop_duplicates('id_iklan').reset_index(drop=True)

    df.insert(3, 'komplek', df[['fasilitas']].fillna('none').apply(lambda z: 'ya' if any([x in z.fasilitas.lower() for x in ['lapangan','gym','jogging','playground','one gate system']]) else 'tidak', axis = 1))
    df.insert(1, 'Kecamatan', df.alamat.str.split(',').str[0])
    df.insert(2, 'Kota', df.alamat.str.split(',').str[1])
    df.drop(columns=['alamat'], inplace=True)

    df = df[df.Kota == ' Tangerang']
    df.drop(columns = ['Kota'], inplace=True)

    #create a function to categorize multiple 'perabot' conditions into more general category
    def perabot(kondisi):
        kondisi = str(kondisi)
        if kondisi.lower() in ['unfurnished','butuh renovasi']:
            return 'Unfurnished'
        elif kondisi.lower() in ['furnished','bagus','bagus sekali','baru','sudah renovasi','semi furnished']:
            return 'Furnished'
        else:
            return np.nan

    df['kondisi_perabotan'] = df['kondisi_perabotan'].map(lambda x: perabot(x))

    df = df[df['tipe_properti'] == 'Rumah']

    #extract prices from column 'harga' into integer
    def price_extract(price):
        if "Triliun" in price:
            numbers = re.findall(r'\d+\.\d+|\d+', price)
            numbers = float(".".join(numbers))
            return int(numbers * 1000000)
        elif "Miliar" in price:
            numbers = re.findall(r'\d+\.\d+|\d+', price)
            numbers = float(".".join(numbers))
            return int(numbers * 1000)
        elif "Juta" in price:
            numbers = re.findall(r'\d+\.\d+|\d+', price)
            numbers = float(".".join(numbers))
            return int(numbers * 1)      

    df.insert(1, 'price',  df[['harga']].apply(lambda x: price_extract(x.harga), axis = 1))
    df.drop(columns = ['harga'], inplace = True)

    df.insert(3, 'ID',  df['id_iklan'])

    df = df[df.sertifikat == 'SHM - Sertifikat Hak Milik']
    df = df[~df['luas_tanah'].isna()].reset_index(drop=True)
    df['luas_bangunan'] = df['luas_bangunan'].fillna(df['luas_tanah'])

    condition = ((df['daya_listrik'] == 'Lainnya Watt') | (df['daya_listrik'] == 'lainnya Watt'))
    df.loc[condition, 'daya_listrik'] = np.nan

    df['daya_listrik'] = df['daya_listrik'].str.replace('Watt','').astype('Float64')
    df['luas_bangunan'] = df['luas_bangunan'].str.replace('m²','').astype(int)
    df['luas_tanah'] = df['luas_tanah'].str.replace('m²','').astype(int)

    df['avg_bangunan'] = df.groupby('daya_listrik')['luas_bangunan'].transform('median').fillna(df['luas_bangunan'])
    listrik_impute = pd.DataFrame(KNNImputer(n_neighbors=1).fit_transform(df[['daya_listrik','avg_bangunan']]))
    df['daya_listrik'] = listrik_impute[0]
    df.drop(columns=['avg_bangunan'], inplace=True)

    value_mapping = {
        'Unfurnished': 1,
        'Furnished' : 2,
    }

    reverse_mapping = {
        1.0 : 'Unfurnished',
        2.0 : 'Furnished',
    }

    for kec in df.Kecamatan.unique() :
        df1 = df[df.Kecamatan == kec][['ID','kondisi_perabotan','luas_tanah','luas_bangunan','price']].copy().reset_index(drop=True)
        df1['harga/m2'] = df1.price / (df1['luas_tanah'] + df1['luas_bangunan'])
        df1['hargagroup'] = df1.groupby(['kondisi_perabotan'])['harga/m2'].transform('median').fillna(df1['harga/m2'])
        df1['kondisi_perabotan'] = df1['kondisi_perabotan'].map(value_mapping).astype('Int64')
        
        perabot_impute = pd.DataFrame(KNNImputer(n_neighbors=3).fit_transform(df1[['kondisi_perabotan','hargagroup']]))
        df1['kond_perabot'] = perabot_impute[0]

        df.set_index('ID', inplace=True, drop=False)
        df1.set_index('ID', inplace=True, drop=False)

        df['kondisi_perabotan'] = df['kondisi_perabotan'].combine_first(df1['kond_perabot'])

        df.reset_index(drop=True, inplace=True)

    df['kondisi_perabotan'] = df['kondisi_perabotan'].replace(reverse_mapping)

    df = df[['price','Kecamatan','kondisi_perabotan','luas_bangunan','luas_tanah','daya_listrik']]

    df = df.drop_duplicates()
    df = df[(df.price > 150) & (df.price < 10000)]
    df['kondisi_perabotan'] = df['kondisi_perabotan'].map(lambda x: 2 if x == 'Furnished' else 1) 

    return df
