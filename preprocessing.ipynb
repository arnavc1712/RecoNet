{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch.nn as nn\n",
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dir = os.path.join(os.getcwd(),\"data\",\"ml-100k\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'/Users/arnav1712/Desktop/University_Stuff/SWM/RecNet/data/ml-100k'"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_dir"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_arr = []"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(os.path.join(data_dir,\"u.data\")) as f:\n",
    "    data = f.readlines()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "for row in data:\n",
    "    data_arr.append(row.strip().split(\"\\t\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_arr = np.array(data_arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_df = pd.DataFrame(data=data_arr,columns=[\"user_id\",\"item_id\",\"rating\",\"timestamp\"])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_df = data_df.sort_values(['user_id', 'timestamp'], ascending=[1, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "step_size=1\n",
    "max_len = 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>user_id</th>\n",
       "      <th>item_id</th>\n",
       "      <th>rating</th>\n",
       "      <th>timestamp</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>59972</th>\n",
       "      <td>1</td>\n",
       "      <td>168</td>\n",
       "      <td>5</td>\n",
       "      <td>874965478</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>92487</th>\n",
       "      <td>1</td>\n",
       "      <td>172</td>\n",
       "      <td>5</td>\n",
       "      <td>874965478</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>74577</th>\n",
       "      <td>1</td>\n",
       "      <td>165</td>\n",
       "      <td>5</td>\n",
       "      <td>874965518</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48214</th>\n",
       "      <td>1</td>\n",
       "      <td>156</td>\n",
       "      <td>4</td>\n",
       "      <td>874965556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15764</th>\n",
       "      <td>1</td>\n",
       "      <td>196</td>\n",
       "      <td>5</td>\n",
       "      <td>874965677</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      user_id item_id rating  timestamp\n",
       "59972       1     168      5  874965478\n",
       "92487       1     172      5  874965478\n",
       "74577       1     165      5  874965518\n",
       "48214       1     156      4  874965556\n",
       "15764       1     196      5  874965677"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['1', '1', '1', ..., '99', '99', '99'], dtype=object)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "unique_user_id,indices,counts = np.unique(data_df['user_id'].values,\n",
    "                                          return_index=True,\n",
    "                                          return_counts=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sliding_window(item_ids,step_size,max_len):\n",
    "    \n",
    "    for i in range(len(item_ids),0,-step_size):\n",
    "        yield item_ids[max(i-max_len,0):i]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def return_sequences(df,indices,unique_user_id,max_len,step_size):\n",
    "    \n",
    "    for idx,start_index in enumerate(indices):\n",
    "        if idx==len(indices)-1:\n",
    "            break\n",
    "        user_id=unique_user_id[idx]\n",
    "\n",
    "        last_index = indices[idx+1]\n",
    "\n",
    "        \n",
    "        for seq in sliding_window(df['item_id'].values[start_index:last_index],step_size,max_len) :\n",
    "            \n",
    "            yield (user_id,seq)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "seq_arr = []\n",
    "for seq in return_sequences(data_df,indices,unique_user_id,max_len,step_size):\n",
    "    seq_arr.append(seq)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [],
   "source": [
    "!rm sequences.pkl\n",
    "!touch sequences.pkl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open(\"sequences.pkl\",\"wb\") as f:\n",
    "    pickle.dump(seq_arr,f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [],
   "source": [
    "arr = pickle.load(open(\"sequences.pkl\",\"rb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "('1', array(['209', '32', '189', '242', '171', '111', '256', '5', '74', '102'],\n",
       "       dtype=object))"
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "APG",
   "language": "python",
   "name": "apg"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
