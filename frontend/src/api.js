
/*
  payload: json containing the endpoint's parameters
*/
export async function upload(payload) {

  const formData = new FormData();

  Object.keys(payload).forEach(key => {
    formData.append(key, payload[key]);
  });
  
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
