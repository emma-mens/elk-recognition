docker pull openvino/workbench

mkdir -p ~/.workbench

docker run -p 127.0.0.1:5665:5665 \
                --name workbench \
                --volume ~/.workbench:/home/openvino/.workbench
                -d openvino/workbench:latest

# to stop
# docker stop workbench

# to monitor logs
# docker logs workbench
