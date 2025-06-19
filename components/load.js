import useSWR from 'swr'

const fetcher = (url) => fetch(url).then((res) => res.json())

const load = (path1, path2) => {
  const { data: data1, error: error1 } = useSWR(path1, fetcher, {
    refreshInterval: 5000,
  })

  const { data: data2, error: error2 } = useSWR(path2, fetcher, {
    refreshInterval: 5000,
  })

  return {
    data: { mayoral: data1, council: data2 },
    error: { mayoral: error1, council: error2 },
  }
}

export default load
