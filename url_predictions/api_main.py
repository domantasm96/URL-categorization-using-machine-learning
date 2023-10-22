from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from url_predictions.functions import format_url, parse_response, predict_category, read_pickle, scrape_url

app = FastAPI()

words_frequency = read_pickle("frequency_models/word_frequency_2021.pickle")


@app.get("/healthcheck/")
def healthy_condition() -> dict[str, str]:
    return {"status": "online"}


@app.post("/predict/")
async def predict_url(url: str) -> ORJSONResponse:
    url = format_url(url)
    response = [0, scrape_url(url, prediction=True)]
    html_content = parse_response(response)
    results = predict_category(words_frequency, html_content)
    results["response"] = response[1]  # type: ignore
    results["tokens"] = html_content[1]  # type: ignore
    return ORJSONResponse(content=results)
