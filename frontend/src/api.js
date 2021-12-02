
/*
  payload: json containing the endpoint's parameters
*/
export async function upload(payload) {

  const formData = new FormData();

  formData.append("file", payload["file"]);
  formData.append("model", payload["model"]);
  formData.append("extract_faces", payload["extract_faces"]);

  let res = await fetch("BASE_URL/models.age.predict", {
    method: "POST",
    body: formData,
    headers: {
      Accept: "*/*",
      "Accept-Encoding": "gzip, deflate, br",
    },
  }).then((r) => r.json());

  return res;
}

export async function getModels() {

  let res = await fetch("BASE_URL/models.age.list", {
    method: "GET",
    headers: {
      Accept: "*/*",
      "Accept-Encoding": "gzip, deflate, br",
    },
  }).then((r) => r.json());

  //console.log(res.data)
 
  return res;
}
