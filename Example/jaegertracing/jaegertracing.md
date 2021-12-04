# jaegertracing

https://www.jaegertracing.io/docs/1.29/getting-started/#all-in-one

```bash
docker run -d --name jaeger \
  -e COLLECTOR_ZIPKIN_HOST_PORT=:9411 \
  -p 5775:5775/udp \
  -p 6831:6831/udp \
  -p 6832:6832/udp \
  -p 5778:5778 \
  -p 16686:16686 \
  -p 14268:14268 \
  -p 14250:14250 \
  -p 9411:9411 \
  jaegertracing/all-in-one:1.29
```

# 追踪配置

https://docs.bentoml.org/en/latest/guides/tracing.html

```bash
bentoml serve $BENTO_BUNDLE_PATH --config my_config_file.yml



```

When starting a BentoML API model server, provide the path to this config file via the CLI argument –config:

bentoml serve $BENTO_BUNDLE_PATH --config my_config_file.yml After BentoML v0.13.0, user will need to provide the config
file path via environment variable BENTOML_CONFIG:

BENTOML_CONFIG=my_config_file.yml bentoml serve $BENTO_BUNDLE_PATH Similarly when serving with BentoML API server docker
image, assuming you have a my_config_file.yml file ready in current directory:

docker run -v $(PWD):/tmp my-bento-api-server -p 5000:5000 --config /tmp/my_config_file.yml

# after version 0.13.0

docker run -v $(PWD):/tmp -p 5000:5000 -e BENTOML_CONFIG=/tmp/my_config_file.yml my-bento-api-server BentoML has already
implemented basic tracing information for its micro-batching server and model server. If there’s additional tracing that
you’d like to add to your BentoML

model server, you may import
