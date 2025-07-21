#!/bin/bash

# Create IDyOM directory structure
mkdir -p ~/idyom/{db,data/{cache,models,resampling}}

# Install Quicklisp (Common Lisp package manager)
mkdir -p ~/quicklisp
wget http://beta.quicklisp.org/quicklisp.lisp -O ~/quicklisp/quicklisp.lisp
echo | sbcl --load ~/quicklisp/quicklisp.lisp \
     --eval "(quicklisp-quickstart:install :path \"~/quicklisp/\")" \
     --eval "(ql:add-to-init-file)" \
     --eval "(ql:quickload \"quicklisp-slime-helper\")"

# Install IDyOM
wget https://github.com/mtpearce/idyom/archive/refs/heads/master.zip -O idyom.zip
unzip idyom.zip -d ~/quicklisp/local-projects/
mv ~/quicklisp/local-projects/idyom-master ~/quicklisp/local-projects/idyom

# Create or update .sbclrc configuration
cat >> ~/.sbclrc << 'EOF'

;;; Load CLSQL by default
(ql:quickload "clsql")

;;; IDyOM
(defun start-idyom ()
   (defvar *idyom-root* (namestring (merge-pathnames "idyom/" (user-homedir-pathname))))
   (defvar *idyom-message-detail-level* 1)
   (ql:quickload "idyom")
   (clsql:connect (list (namestring (merge-pathnames "idyom/db/database.sqlite" (user-homedir-pathname)))) 
                  :if-exists :old :database-type :sqlite3))
EOF

# Initialize IDyOM database
sbcl --non-interactive \
     --load ~/quicklisp/setup.lisp \
     --eval "(ql:quickload \"clsql\")" \
     --eval "(defun start-idyom () (defvar *idyom-root* (namestring (merge-pathnames \"idyom/\" (user-homedir-pathname)))) (defvar *idyom-message-detail-level* 1) (ql:quickload \"idyom\") (clsql:connect (list (namestring (merge-pathnames \"idyom/db/database.sqlite\" (user-homedir-pathname)))) :if-exists :old :database-type :sqlite3))" \
     --eval "(start-idyom)" \
     --eval "(idyom-db:initialise-database)" \
     --eval "(quit)"

echo "IDyOM setup complete!" 