class DataSource(object):
    def _load_raw_data(self,filename, is_train=True):        
        a = []
        b = []        
        regex = 'train_'
        if not is_train:
            regex = 'test_'        
        with open(filename, 'r') as file:
            for line in file :
                if regex in line:
                    b.append(a)
                    a = [line]
                elif line!='\n':
                    a.append(line)                    
        b.append(a)      
        return b[1:]

    def _create_row(self, sample, is_train=True):        
        d = {}
        d['id'] = sample[0].replace('\n','')
        review = ""
        if is_train:
            for clause in sample[1:-1]:
                review+= clause.replace('\n','').strip()
            d['label'] = int(sample[-1].replace('\n',''))          
        else:         
            for clause in sample[1:]:
                review+= clause.replace('\n','').strip()
        d['review'] = review
        return d
    
    def load_data(self, filename, is_train=True):    
        raw_data = self._load_raw_data(filename, is_train)
        lst = []
        for row in raw_data:
            lst.append(self._create_row(row, is_train))
        return lst