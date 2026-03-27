FROM julia:1.12

WORKDIR /app

ENV FIREBASE_PROJECT_ID=""
ENV GOOGLE_APPLICATION_CREDENTIALS="/secrets/firebase-service-account.json"

COPY Project.toml Manifest.toml server.jl ./
COPY src ./src
COPY test ./test

RUN julia --project=. -e "using Pkg; Pkg.instantiate()"

EXPOSE 8080

CMD ["julia", "--project=.", "server.jl"]
