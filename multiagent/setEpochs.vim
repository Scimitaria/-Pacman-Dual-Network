let g:epochs = !empty($EPOCHS) ? $EPOCHS : 0
:%s/\<self\.epochs\s*=\s*\d\+\>/\="self.epochs = " . g:epochs/g