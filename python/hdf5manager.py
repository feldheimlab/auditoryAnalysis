#!/usr/bin/env python3

'''
Tools for standardized saving/loading a class or dictionary to a .hdf5 file.
Not all formats are supported, feel free to add a set of save/load functions for 
unsupported formats. 
Strings are saved as attributes of the file; lists of strings are saved as tab 
delimited strings; arrays are saved as datasets.  Dicts are saved as a new folder, 
with data saved as numpy datasets.
Useage:
* listing objects in an hdf5 file:
    f = hdf5manager(mypath)
    f.print()
* saving data to file:
    f = hdf5manager(mypath)
    f.save(mydict)
  OR:
    f.save(myClass)
* loading data from file:
    f = hdf5manager(mypath)
    data = f.load()
Authors: Sydney C. Weiser
Date: 2017-01-13
'''

import h5py
import numpy as np
import os
import pickle

def main():
    '''
    If called directly from command line, take argument passed, and try to read 
    contents if it's an .hdf5 file.
    '''
    import argparse

    print('\nHDF5 Manager\n-----------------------')

    ap = argparse.ArgumentParser()
    ap.add_argument('file', type = argparse.FileType('r'), nargs='+',
        help = 'path to the hdf5 file(s)')
    ap.add_argument('-e', '--extract', type=str, nargs='+', 
        help='key(s) to be extracted')
    ap.add_argument('-m', '--merge', type=argparse.FileType('r'), 
        help='merges keys in merge file into main file(s).')
    ap.add_argument('-c', '--copy', action='store_true', 
        help='make copy of hdf5 file')
    ap.add_argument('-d', '--delete', type=str, nargs='+', help='delete key')
    ap.add_argument('-r', '--rename', type=str, nargs=2, help='rename key')
    ap.add_argument('-i', '--ignore', type=str, 
        help='key to ignore while loading.  For use with copy.')
    ap.add_argument('--read', type=str, nargs='+',
        help='key(s) to read to terminal.')


    args = vars(ap.parse_args())

    if len(args['file']) == 1:        
        path = args['file'][0].name

        assert path.endswith('.hdf5'), 'Not a valid hdf5 file.\nExiting.\n'

        print('Found hdf5 file:', path)
        f = hdf5manager(path, verbose=True)
        # f.print()

        if args['extract'] is not None:
            print('extracting keys:', ', '.join(args['extract']), '\n')

            for key in args['extract']:
                assert key in f.keys(), '{0} was not a valid key!'.format(key)

            loaded = f.load(args['extract'])

            if type(loaded) is not dict:
                loaded = {args['extract'][0]:loaded}

            newpath = f.path.replace('.hdf5', 
                '_extract_{0}.hdf5'.format('-'.join(args['extract'])))

            print('new path:', newpath)

            hdf5manager(newpath).save(loaded)

        elif args['merge'] is not None:
            mergepath = args['merge'].name
            assert mergepath.endswith('.hdf5'), 'merge file was not valid'
            print('merging hdf5 file:', mergepath)

            mergedict = hdf5manager(mergepath).load()
            # print(mergedict)

            for key in mergedict.keys():
                print(key)

                if key in f.keys():
                    print('found in key, are you sure you want to merge? (y/n)')

                    loop = True

                    while loop:
                        response = input().lower().strip()
                        if (response == 'y') | (response == 'yes'):
                            loop = False
                            f.save({key: mergedict[key]})
                        elif (response == 'n') | (response == 'no'):
                            print('not saving', key)
                            loop = False
                        else:
                            print('invalid answer!')

                else:
                    print(key, 'not in main file.  No merge conflicts')
                    f.save({key: mergedict[key]})

        elif args['copy']:
            ignore = args['ignore']

            if ignore is not None:
                assert ignore in f.keys(), '{0} not a valid key!'.format(ignore)

            data = f.load(ignore=ignore)

            newpath = f.path.replace('.hdf5', '_copy.hdf5')
            g = hdf5manager(newpath)
            g.save(data)

        elif args['delete']:

            f.delete(args['delete'])

            print('Note: deleting keys from hdf5 file may not free up space.')
            print('Make a copy with --copy command to free up space.')

        elif args['rename']:
            print('renaming', args['rename'][0], 'to', args['rename'][1], '\n')

            key = args['rename'][0]
            assert key in f.keys(), \
                    'key was not valid: ' + key
            f.verbose=False

            data = f.load(key)
            print('data loaded:', data)

            f.save({args['rename'][1] : data})

            f.open()
            try:
                del f.f[key]
            except:
                del f.f.attrs[key]
            f.close()
            f.print()

        elif args['read']:
            f.verbose=False
            for key in args['read']:
                if key in f.keys():
                    print('key found:', key)
                    print(key+':', f.load(key))
                else:
                    print('key not found:', key)

        else:
            print('no additional commands found')

    elif len(args['file']) > 1:
        pathlist = [file.name for file in args['file']]

        print('\nFound multiple files:')
        [print('\t', path) for path in pathlist]
        print('')

        if args['extract'] is not None:
            print('extracting keys:', ', '.join(args['extract']), '\n')

            data = loadKeys(pathlist, args['extract'])

            directory = os.path.dirname(pathlist[0])
            directory = os.path.abspath(directory)
            filename = 'hdf5extract-' + '-'.join(args['extract'])+'.hdf5'

            path = os.path.join(directory, filename)
            print('path', path)

            f = hdf5manager(path)
            f.save(data)

        elif args['merge'] is not None:
            print('testing!')

            mergepath = args['merge'].name
            assert mergepath.endswith('.hdf5'), 'merge file was not valid'
            print('merging hdf5 file:', mergepath)

            mergedict = hdf5manager(mergepath).load()
            key = list(mergedict.keys())[0]
            # get a random key, test if subdict structure or one key extracted

            if type(mergedict[key]) is dict: 
                subdict=True
            else:
                subdict=False
                mergekey = os.path.basename(mergepath).replace('.hdf5', ''
                                    ).replace('hdf5extract-', '')

            for path in pathlist:
                name = os.path.basename(path).replace('.hdf5', '').replace('_ica', '')
                print('Merging file:', name)

                if name in mergedict.keys():
                    f = hdf5manager(path, verbose=False)

                    if subdict:
                        conflict = []
                        for key in mergedict[name]:
                            if key in f.keys():
                                conflict.append(key)

                        if len(conflict) > 0:
                            print('replace old dict with new?')

                            print('\tOriginal:')
                            print('\t', f.load(conflict))
                            print('\n\tNew:')
                            print('\t', mergedict[name])
                            print('\n\treplace?')

                            loop = True
                            while loop:
                                response = input().lower().strip()
                                if (response == 'y') | (response == 'yes'):
                                    loop = False
                                    f.save(mergedict[name])
                                elif (response == 'n') | (response == 'no'):
                                    print('not saving')
                                    loop = False
                                else:
                                    print('invalid answer!')
                        else:
                            f.save(mergedict[name])


                    else:
                        if mergekey in f.keys():
                            print(mergekey, 'was in original file.')
                            print('Are you sure you want to replace this?')

                            print('\tOriginal:')
                            print('\t', f.load(mergekey))
                            print('\n\tNew:')
                            print('\t', mergedict[name])
                            print('\n\treplace?')

                            loop = True
                            while loop:
                                response = input().lower().strip()
                                if (response == 'y') | (response == 'yes'):
                                    loop = False
                                    f.save({mergekey: mergedict[name]})
                                elif (response == 'n') | (response == 'no'):
                                    print('not saving', key)
                                    loop = False
                                else:
                                    print('invalid answer!')
                        else:
                            f.save({mergekey: mergedict[name]})

                else:
                    print(name, 'not found in merge dictionary')
                    print('skipping...')

        elif args['read']:
            for path in pathlist:
                print('\n', path)
                f = hdf5manager(path, verbose=False)
                for key in args['read']:
                    if key in f.keys():
                        print('key found:', key)
                        print(key+':', f.load(key))
                    else:
                        print('key not found:', key)

        else: print('Command not defined for multiple files.')

    else: print('No hdf5file found')



def loadKeys(pathlist, keys=None):
    '''
    If no keys are passed in, all are loaded
    '''
    if type(pathlist) is str:
        pathlist = [pathlist]
    data = dict()

    for path in pathlist:
        assert os.path.isfile(path), 'File was invalid: {0}'.format(path)
        name = os.path.basename(path).replace('.hdf5', '').replace('_ica', '')
        print('Loading File:', name)

        try:
            f = hdf5manager(path, create=False)
            filedata = f.load(keys)
        except:
            filedata = None
        
        data[name] = filedata

    return data

class hdf5manager:
    def __init__(self, path, verbose=False, create=True):

        assert (path.endswith('.hdf5') | path.endswith('.mat'))
        path = os.path.abspath(path)

        if not os.path.isfile(path) and create:
            # Create the file
            print('Creating file at:', path)
            f = h5py.File(path, 'w')
            f.close()
        else:
            assert os.path.isfile(path), 'File does not exist'

        self.path = path
        self.verbose = verbose

        if verbose:
            self.print()

    def print(self):
        path = self.path
        print()

        # If not saving or loading, open the file to read it
        if not hasattr(self, 'f'):
            print('Opening File to read...')
            f = h5py.File(path, 'r')
        else:
            f = self.f
    
        if len(list(f.keys())) > 0:
            print('{0} has the following keys:'.format(path))
            for file in f.keys():
                print('\t-',file)
        else:
            print('{0} has no keys.'.format(path))

        if len(list(f.attrs)) > 0:
            print('{0} has the following attributes:'.format(path))
            for attribute in f.attrs:
                print('\t-', attribute)
        else:
            print('{0} has no attributes.'.format(path))

        # If not saving or loading, close the file after finished
        if not hasattr(self, 'f'):
            print('Closing file...')
            f.close()
        print()

    def keys(self):
        # If not saving or loading, open the file to read it
        if not hasattr(self, 'f'):
            f = h5py.File(self.path, 'r')
        else:
            f = self.f

        keys = [key for key in f.attrs]
        keys.extend([key for key in f.keys()])

        if not hasattr(self, 'f'):
            f.close()

        return keys

    def open(self):
        path = self.path
        verbose = self.verbose

        f = h5py.File(path, 'a')
        self.f = f

        self.print() # print all variables

        if verbose:
            print('File is now open for manual accessing.\n'
                'To access a file handle, assign hdf5manager.f.[key] to a handle'
                ' and pull slices: \n'
                '\t slice = np.array(handle[0,:,1:6])\n'
                'It is also possible to write to a file this way\n'
                '\t handle[0,:,1:6] = np.zeros(x,y,z)\n')

    def close(self):
        self.f.close()
        del self.f

    def load(self, target=None, ignore=None):
        path = self.path
        verbose = self.verbose

        def loadDict(f, key):
            # Load dict to key from its folder
            if verbose: print('\t\t-', 'loading', key, 'from file...')
            g = f[key]

            if verbose: print('\t\t-', key, 'has the following keys:')
            if verbose: print('\t\t  ', ', '.join([gkey for gkey in g.keys()]))

            data = {}
            if g.keys().__len__() > 0:
                for gkey in g.keys():
                    if type(g[gkey]) is h5py.Group:
                        data[gkey] = loadDict(g, gkey)
                    elif type(g[gkey]) is h5py.Dataset:
                        if verbose: print('\t\t-', 'loading', key, 'from file...')
                        data[gkey] = np.array(g[gkey])
                    else:
                        if verbose: print('key was of unknown type', type(gkey))

            if verbose: print('\t\t-', key, 'has the following attributes:')
            if verbose: print('\t\t  ', ', '.join([gkey for gkey in g.attrs]))

            for gkey in g.attrs:
                if verbose: print('\t\t\t', gkey+';', type(g.attrs[gkey]).__name__)
                if verbose: print('\t\t\t-', 'loading', gkey, 'from file...')
                if type(g.attrs[gkey]) is str:
                    data[gkey] = g.attrs[gkey]
                elif type(g.attrs[gkey] is np.void):
                    out = g.attrs[gkey]
                    data[gkey] = pickle.loads(out.tostring())
                else:
                    print('INVALID TYPE:', type(g.attrs[gkey]))

            return data

        f = h5py.File(path, 'a') # Open file for access
        self.f = f # set to variable so other functions know file is open

        if target is None:
            if verbose: print('No target key specified; loading all datasets')
            keys = f.keys()
            attrs = f.attrs
        else:
            assert (type(target) is str) or (type(target) is list), 'invalid target'
            if type(target) is str:
                target = [target]

            keys = []
            attrs = []

            for item in target:

                if (type(item) is str) & (item in f.keys()):
                    if verbose: print('Target key found:', item)
                    keys.append(item)

                elif (type(item) is str) & (item in f.attrs):
                    if verbose: print('Target attribute found:', item)
                    attrs.append(item)

                else:
                    print('Target was not valid:', item)

        if verbose: print('\nLoading datasets from hdf5 file:')
        data = {}
        for key in keys:
            if verbose: print('\t', key + ';', type(f[key]).__name__)

            if key == ignore:
                if verbose: print('\t\t- ignoring key:', key)
            else:
                if type(f[key]) is h5py.Group:
                    data[key] = loadDict(f, key)
                elif type(f[key]) is h5py.Dataset:
                    if verbose: print('\t\t-', 'loading', key, 'from file...')

                    if f[key].dtype.type is np.void:
                        data[key] = pickle.loads(np.array(f[key]).tostring())
                    else:
                        data[key] = np.array(f[key])
                else:
                    if verbose: print('\t\t- attribute was unsupported type:', 
                        type(f[key]).__name__)

        for key in attrs:
            if verbose: print('\t', key + ';', type(f.attrs[key]).__name__)

            if key == ignore:
                if verbose: print('ignoring attribute:', key)
            else:
                if verbose: print('\t\t-', 'loading', key, 'from file...')
                if type(f.attrs[key]) is str:
                    data[key] = f.attrs[key]
                elif type(f.attrs[key] is np.void):
                    out = f.attrs[key]
                    data[key] = pickle.loads(out.tostring())

        if verbose: print('Keys extracted from file:')
        if verbose: print('\t', ', '.join([key for key in data.keys()]))
        if verbose: print('\n\n')

        del self.f
        f.close()

        if (type(target) is list) and (len(target) == 1):
            data = data[target[0]]

        return data

    def delete(self, target):
        if type(target) is not list:
            target = [target]

        f = h5py.File(self.path, 'a') # Open file for access
        self.f = f # set to variable so other functions know file is open
        verbose=self.verbose

        for key in target:
            if key in self.keys():
                if verbose: print('key found:', key)

                try:
                    del f[key]
                except:
                    del f.attrs[key]
            else:
                if verbose: print('key not found:', key)

        del self.f
        f.close()

    def save(self, data):
        # data is a class file or dict of keys/data
        path = self.path
        verbose = self.verbose

        '''
        Saves a class or dict to hdf5 file.
        Note that lists of numbers are not supported, only np arrays or 
        lists of strings.
        '''

        # Functions to save each type of data:
        # --------------------------------------------------------------

        def saveDict(f, fdict, key):
            # Write dict to key as its own folder
            if verbose: print('\t\t-', 'writing', key, 'to file...')

            # Delete if it exists
            if key in f:
                if verbose: print('\t\t-', 'Removing', key, 'from file')
                del f[key]

            g = f.create_group(key)
            data_d = fdict

            for dkey in fdict:

                if (type(fdict[dkey]) is str):
                    saveString(g, fdict[dkey], dkey)
                elif type(fdict[dkey]) is np.ndarray:
                    saveArray(g, fdict[dkey], dkey)
                elif type(fdict[dkey]) is dict:
                    saveDict(g, fdict[dkey], dkey)
                else:
                    if verbose: print('\t\t- attribute was unsupported type:', 
                        type(fdict[dkey]).__name__)
                    if verbose:
                        print('\t\tAttempting to save pickle dump of object')
                    try:
                        saveOther(g, fdict[dkey], dkey)
                        if verbose: print('\t\tSaved succesfully!')
                    except:
                        if verbose: print('\t\tFailed..')

            if verbose: print('\t\t-', key, 'has the following keys:')
            if verbose: print('\t\t  ', ', '.join([dkey for dkey in g.keys()]))

            if verbose: print('\t\t-', key, 'has the following attributes:')
            if verbose: print('\t\t  ', ', '.join([dkey for dkey in g.attrs]))            

        def saveString(f, string, key):
            # Write all strings as attributes of the dataset
            if verbose: print('\t\t-', 'writing', key, 'to file...')
            f.attrs[key] = string

        def saveArray(f, array, key):
            # Check if key exists, and if entry is the same as existing value
            if key in f.keys():
                if (not np.array_equal(array, 
                        f[key])):
                    if verbose: print('\t\t-',key,'in saved file is inconsistent '
                        'with current version')
                    if verbose: print('\t\t-', 'deleting', key, 'from file')
                    del f[key]
                    if verbose: print('\t\t-', 'writing', key, 'to file...')
                    f.create_dataset(key, data=array, chunks=None)
                else:
                    if verbose: print('\t\t-',key,'in saved file is the same as '
                        'the current version')
            else:
                if verbose: print('\t\t-', 'writing', key, 'to file...')
                f.create_dataset(key, data=array, chunks=None)

        def saveOther(f, obj, key):
            # Compress to bytestring using pickle, save similar to string
            # Write all strings as attributes of the dataset
            if verbose: print('\t\t-', 'writing', key, 'to file...')

            bstring = np.void(pickle.dumps(obj))
            try:
                f.attrs[key] = bstring
            except RuntimeError:
                if verbose: print('\t\t\tEncountered RuntimeError')
                if verbose: print('\t\t\tSaving pickle dump as data...')
                if key in f.keys():
                    if verbose: print('Deleting previous copy of', key)
                    del f[key]
                f[key] = bstring


        # Check input data type, open file:
        # --------------------------------------------------------------

        # If data is not a dictionary, assume 
        if type(data) is not dict:
            # Get dictionary of all keys in class type
            data = data.__dict__

        if verbose:
            print('Attributes found in data file:')
            for key in data.keys():
                print('\t', key, ':', type(data[key]))

        f = h5py.File(path, 'a')
        self.f = f

        if verbose:
            self.print()

        # Loop through keys and save them in hdf5 file:
        # --------------------------------------------------------------
        if verbose: print('\nSaving class attributes:')
        for key in data.keys():
            if verbose: print('\t', key + ';', type(data[key]).__name__)
            if (type(data[key]) is str):
                saveString(f, data[key], key)
            elif type(data[key]) is np.ndarray:
                saveArray(f, data[key], key)
            elif type(data[key]) is dict:
                saveDict(f, data[key], key)
            else:
                if verbose: print('\t\t- attribute was unsupported type:', 
                    type(data[key]).__name__)
                if verbose: print('\t\tAttempting to save pickle dump of object')
                try:
                    saveOther(f, data[key], key)
                    if verbose: print('\t\tSaved succesfully!')
                except:
                    print('\t\tFailed..')
                    print('\t\t\t', key, 'did not save')
                    print('\t\t\t', 'type:', type(data[key]).__name__)
        
        self.print()

        del self.f
        f.close()

if __name__ == '__main__':
    main()